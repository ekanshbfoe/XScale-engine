"""
XScale Engine — FastAPI Bridge
═══════════════════════════════
Runs inside Google Colab. Creates a Cloudflare Tunnel and exposes
a FastAPI server for the mobile app to communicate with.

Endpoints:
  POST /upscale  — start processing
  GET  /status   — current engine state
  POST /stop     — graceful shutdown

On startup, the tunnel URL is written to Firebase so the app can
discover the engine.
"""

import os
import re
import json
import time
import subprocess
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ─── Firebase Admin SDK ───────────────────────────────────

import firebase_admin
from firebase_admin import credentials, db as firebase_db

# ─── Configuration ────────────────────────────────────────

# The user's Firebase credentials JSON (uploaded to Colab)
FIREBASE_CRED_PATH = os.environ.get(
    "FIREBASE_CRED_PATH", "/content/firebase-service-account.json"
)
FIREBASE_DB_URL = os.environ.get(
    "FIREBASE_DB_URL", "https://xscale-1eda4-default-rtdb.asia-southeast1.firebasedatabase.app"
)
USER_UID = os.environ.get("USER_UID", "")  # Set by the notebook

IDLE_TIMEOUT_SECONDS = 60 * 60  # 60 minutes
POST_PROCESSING_SHUTDOWN_DELAY = 10  # seconds

# ─── State ────────────────────────────────────────────────

engine_state = {
    "status": "booting",       # idle | booting | processing | complete
    "progress": 0,             # 0–100
    "tunnel_url": "",
    "error": None,
}

last_heartbeat = time.time()

# ─── Firebase Setup ───────────────────────────────────────

def init_firebase():
    """Initialize Firebase Admin SDK."""
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})


def update_firebase(field: str, value):
    """Write a value to users/{uid}/{field} in Firebase RTDB."""
    if not USER_UID:
        return
    ref = firebase_db.reference(f"users/{USER_UID}/{field}")
    ref.set(value)


def update_firebase_status(status: str, progress: int = 0):
    """Update both status and progress in Firebase."""
    engine_state["status"] = status
    engine_state["progress"] = progress
    update_firebase("engine_status", status)
    update_firebase("current_progress", progress)


# ─── Cloudflare Tunnel ────────────────────────────────────

def start_tunnel(port: int = 8000) -> str:
    """
    Start cloudflared tunnel and extract the public URL.
    Returns the tunnel URL.
    """
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    url = ""
    # Read output lines to find the tunnel URL
    for line in iter(process.stdout.readline, ""):
        match = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", line)
        if match:
            url = match.group(1)
            break

    if not url:
        raise RuntimeError("Failed to extract Cloudflare tunnel URL")

    engine_state["tunnel_url"] = url
    update_firebase("tunnel_url", url)
    return url


# ─── Idle Watchdog ────────────────────────────────────────

def idle_watchdog():
    """Background thread that checks for idle timeout."""
    global last_heartbeat
    while True:
        time.sleep(30)
        elapsed = time.time() - last_heartbeat
        if elapsed > IDLE_TIMEOUT_SECONDS:
            print(f"[XScale] Idle timeout ({elapsed:.0f}s). Shutting down.")
            graceful_shutdown()
            break


def graceful_shutdown():
    """Disconnect the Colab runtime."""
    update_firebase_status("idle", 0)
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as e:
        print(f"[XScale] Shutdown error: {e}")


# ─── FastAPI App ──────────────────────────────────────────

app = FastAPI(title="XScale Engine", version="1.0.0")


class UpscaleRequest(BaseModel):
    file_name: str
    file_id: str = ""  # Google Drive file ID (preferred over name search)
    scale_factor: float = 2.0
    model_type: str = "realistic"  # "realistic" or "anime"


@app.get("/status")
async def get_status():
    return {
        "status": engine_state["status"],
        "progress": engine_state["progress"],
        "error": engine_state["error"],
    }


@app.post("/upscale")
async def start_upscale(req: UpscaleRequest):
    """Trigger the video upscaling pipeline."""
    if engine_state["status"] == "processing":
        raise HTTPException(status_code=409, detail="Already processing a video")

    # Run processing in background thread
    thread = threading.Thread(
        target=_run_upscale,
        args=(req.file_name, req.file_id, req.scale_factor, req.model_type),
        daemon=True,
    )
    thread.start()

    return {"message": "Processing started", "file": req.file_name}


@app.post("/stop")
async def stop_engine():
    """Gracefully stop the engine."""
    update_firebase_status("idle", 0)
    # Schedule shutdown after a short delay
    threading.Thread(target=lambda: (time.sleep(2), graceful_shutdown()), daemon=True).start()
    return {"message": "Shutting down"}


@app.post("/heartbeat")
async def receive_heartbeat():
    """Update the last heartbeat timestamp."""
    global last_heartbeat
    last_heartbeat = time.time()
    return {"ok": True}


# ─── Processing Pipeline ─────────────────────────────────

def _run_upscale(file_name: str, file_id: str, scale_factor: float, model_type: str):
    """
    Main processing pipeline (runs in background thread).
    1. Download video from Drive
    2. Run AI upscaling
    3. Upload result back to Drive
    4. Update Firebase status
    """
    try:
        update_firebase_status("processing", 0)

        # Import processor module
        from processor import upscale_video
        from cleanup import auto_purge_drive

        # Paths
        input_path = f"/content/temp_in/{file_name}"
        output_name = file_name.rsplit(".", 1)[0] + "_4k.mp4"
        output_path = f"/content/temp_out/{output_name}"

        # Ensure directories exist
        os.makedirs("/content/temp_in", exist_ok=True)
        os.makedirs("/content/temp_out", exist_ok=True)

        # Download from Google Drive (by file ID)
        update_firebase_status("processing", 5)
        _download_from_drive(file_id, input_path)

        # Run upscaling
        update_firebase_status("processing", 10)
        upscale_video(
            input_path=input_path,
            output_path=output_path,
            scale_factor=scale_factor,
            model_type=model_type,
            progress_callback=lambda p: update_firebase_status(
                "processing", 10 + int(p * 0.8)
            ),
        )

        # Upload result back to Drive
        update_firebase_status("processing", 90)
        _upload_to_drive(output_path, output_name)

        # Run cleanup
        update_firebase_status("processing", 95)
        auto_purge_drive()

        # Done!
        update_firebase_status("complete", 100)

        # Auto-shutdown after delay
        time.sleep(POST_PROCESSING_SHUTDOWN_DELAY)
        graceful_shutdown()

    except Exception as e:
        engine_state["error"] = str(e)
        update_firebase("engine_status", "idle")
        print(f"[XScale] Processing error: {e}")


def _get_drive_service():
    """Get an authenticated Google Drive API service using the service account."""
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_file(
        FIREBASE_CRED_PATH, scopes=SCOPES
    )
    return build('drive', 'v3', credentials=creds)


def _find_file_in_drive(service, file_name: str, folder_name: str = "temp_in"):
    """Find a file by name in /XScale/{folder_name}/. Returns file ID or None."""
    # First find the XScale folder
    results = service.files().list(
        q="name='XScale' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id)"
    ).execute()
    xscale_folders = results.get('files', [])
    if not xscale_folders:
        return None

    xscale_id = xscale_folders[0]['id']

    # Find the subfolder
    results = service.files().list(
        q=f"name='{folder_name}' and '{xscale_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id)"
    ).execute()
    sub_folders = results.get('files', [])
    if not sub_folders:
        return None

    sub_id = sub_folders[0]['id']

    # Find the file
    results = service.files().list(
        q=f"name='{file_name}' and '{sub_id}' in parents and trashed=false",
        fields="files(id,name)"
    ).execute()
    files = results.get('files', [])
    return files[0]['id'] if files else None


def _find_or_create_folder(service, name: str, parent_id: str = None):
    """Find a folder by name under parent, or create it. Returns folder ID."""
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    else:
        q += " and 'root' in parents"

    results = service.files().list(q=q, fields="files(id)").execute()
    folders = results.get('files', [])
    if folders:
        return folders[0]['id']

    # Create the folder
    metadata = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder',
    }
    if parent_id:
        metadata['parents'] = [parent_id]

    folder = service.files().create(body=metadata, fields='id').execute()
    return folder['id']


def _download_from_drive(file_id: str, local_path: str):
    """Download a file from Google Drive by file ID via REST API."""
    from googleapiclient.http import MediaIoBaseDownload

    print(f"[XScale] Downloading file {file_id} from Drive...")
    service = _get_drive_service()

    request = service.files().get_media(fileId=file_id)
    with open(local_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"[XScale] Download progress: {int(status.progress() * 100)}%")

    print(f"[XScale] Downloaded to {local_path}")


def _upload_to_drive(local_path: str, file_name: str):
    """Upload a file to Google Drive /XScale/temp_out/ via REST API."""
    from googleapiclient.http import MediaFileUpload

    print(f"[XScale] Uploading {file_name} to Drive...")
    service = _get_drive_service()

    # Ensure folder structure exists
    xscale_id = _find_or_create_folder(service, "XScale")
    temp_out_id = _find_or_create_folder(service, "temp_out", xscale_id)

    file_metadata = {
        'name': file_name,
        'parents': [temp_out_id],
    }
    media = MediaFileUpload(local_path, mimetype='video/mp4', resumable=True)

    request = service.files().create(
        body=file_metadata, media_body=media, fields='id,name'
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"[XScale] Upload progress: {int(status.progress() * 100)}%")

    print(f"[XScale] Uploaded! File ID: {response['id']}")


# ─── Main Entry Point ────────────────────────────────────

def start_engine(user_uid: str, port: int = 8000):
    """
    Call this from the Colab notebook to start everything.
    """
    global USER_UID
    USER_UID = user_uid

    # Ensure Google API client library is installed
    try:
        from googleapiclient.discovery import build
    except ImportError:
        print("[XScale] Installing google-api-python-client...")
        subprocess.run(["pip", "install", "-q", "google-api-python-client"], check=True)

    # 1. Initialize Firebase
    init_firebase()
    update_firebase_status("booting", 0)

    # 2. Start Cloudflare tunnel
    tunnel_url = start_tunnel(port)
    print(f"[XScale] Tunnel URL: {tunnel_url}")

    # 3. Start idle watchdog
    watchdog_thread = threading.Thread(target=idle_watchdog, daemon=True)
    watchdog_thread.start()

    # 4. Mark as ready
    update_firebase_status("idle", 0)

    # 5. Start FastAPI server in background thread (Colab-compatible)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    print(f"[XScale] ✅ Engine ready! Listening on port {port}")
    print(f"[XScale] 🔗 Public URL: {tunnel_url}")

    # Keep the cell alive
    try:
        while server_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("[XScale] Shutting down...")
        server.should_exit = True
