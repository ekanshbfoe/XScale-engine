"""
XScale Engine — Drive Cleanup (5GB Auto-Purge)
════════════════════════════════════════════════
FIFO purge of /XScale/temp_in/ and /XScale/temp_out/ in Google Drive
when total size exceeds 5 GB.

Uses Google Drive REST API — no drive.mount() needed.
"""

import os

MAX_STORAGE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB


def _get_drive_service():
    """Get an authenticated Google Drive API service using the service account."""
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    cred_path = os.environ.get("FIREBASE_CRED_PATH", "/content/firebase-creds.json")
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_file(cred_path, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)


def _find_folder(service, name, parent_id=None):
    """Find a folder by name. Returns folder ID or None."""
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    else:
        q += " and 'root' in parents"

    results = service.files().list(q=q, fields="files(id)").execute()
    folders = results.get('files', [])
    return folders[0]['id'] if folders else None


def _list_files_in_folder(service, folder_id):
    """List all files in a folder with size and creation time. Returns list of dicts."""
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false and mimeType!='application/vnd.google-apps.folder'",
        fields="files(id,name,size,createdTime)",
        orderBy="createdTime"
    ).execute()
    files = results.get('files', [])
    return [
        {
            'id': f['id'],
            'name': f['name'],
            'size': int(f.get('size', 0)),
            'createdTime': f.get('createdTime', ''),
        }
        for f in files
    ]


def auto_purge_drive() -> int:
    """
    FIFO purge: if total storage in /XScale/ exceeds 5 GB,
    delete the oldest files until we're under the limit.

    Returns the number of files deleted.
    """
    service = _get_drive_service()

    # Find XScale folder
    xscale_id = _find_folder(service, "XScale")
    if not xscale_id:
        print("[XScale] No XScale folder found — nothing to purge")
        return 0

    # Find subfolders
    temp_in_id = _find_folder(service, "temp_in", xscale_id)
    temp_out_id = _find_folder(service, "temp_out", xscale_id)

    # List all files
    all_files = []
    if temp_in_id:
        all_files.extend(_list_files_in_folder(service, temp_in_id))
    if temp_out_id:
        all_files.extend(_list_files_in_folder(service, temp_out_id))

    # Sort by creation time (oldest first)
    all_files.sort(key=lambda x: x['createdTime'])

    total = sum(f['size'] for f in all_files)

    if total <= MAX_STORAGE_BYTES:
        print(
            f"[XScale] Storage OK: {total / (1024**3):.2f} GB / "
            f"{MAX_STORAGE_BYTES / (1024**3):.0f} GB"
        )
        return 0

    print(
        f"[XScale] Storage exceeded: {total / (1024**3):.2f} GB / "
        f"{MAX_STORAGE_BYTES / (1024**3):.0f} GB — purging..."
    )

    deleted_count = 0

    for f in all_files:
        if total <= MAX_STORAGE_BYTES:
            break

        try:
            service.files().delete(fileId=f['id']).execute()
            total -= f['size']
            deleted_count += 1
            print(f"[XScale] Deleted: {f['name']} ({f['size'] / (1024**2):.1f} MB)")
        except Exception as e:
            print(f"[XScale] Could not delete {f['name']}: {e}")

    print(
        f"[XScale] Purged {deleted_count} files. "
        f"New total: {total / (1024**3):.2f} GB"
    )
    return deleted_count


if __name__ == "__main__":
    deleted = auto_purge_drive()
    print(f"Deleted {deleted} files")
