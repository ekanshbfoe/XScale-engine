"""
XScale Engine — AI Video Processor
════════════════════════════════════
Handles the actual AI upscaling pipeline:
  1. Extract frames with FFmpeg
  2. Upscale each frame with Real-ESRGAN or Real-CUGAN
  3. Reassemble into H.265 video with FFmpeg
"""

import os
import glob
import subprocess
from typing import Callable, Optional

# ─── Model paths (downloaded in notebook setup) ───────────

MODELS = {
    "realistic": {
        "name": "Real-ESRGAN",
        "binary": "/content/realesrgan-ncnn-vulkan",
        "model_name": "realesrgan-x4plus",
    },
    "anime": {
        "name": "Real-CUGAN",
        "binary": "/content/realcugan-ncnn-vulkan",
        "model_name": "models-pro",
    },
}


def upscale_video(
    input_path: str,
    output_path: str,
    scale_factor: float = 2.0,
    model_type: str = "realistic",
    progress_callback: Optional[Callable[[float], None]] = None,
):
    """
    Full pipeline: extract frames → upscale → reassemble.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        scale_factor: Upscale factor (1.0–4.0)
        model_type: "realistic" or "anime"
        progress_callback: Called with 0.0–1.0 as progress
    """
    # Resolve scale to integer for the AI binary
    scale = max(1, min(4, int(scale_factor)))

    model = MODELS.get(model_type, MODELS["realistic"])
    frames_dir = "/content/temp_frames"
    upscaled_dir = "/content/temp_upscaled"

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(upscaled_dir, exist_ok=True)

    try:
        # ── Step 1: Extract frames ──────────────────────
        if progress_callback:
            progress_callback(0.0)

        print(f"[XScale] Extracting frames from {input_path}")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-qscale:v", "2",
                f"{frames_dir}/frame_%06d.png",
            ],
            check=True,
            capture_output=True,
        )

        # Get frame count
        frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
        total_frames = len(frame_files)

        if total_frames == 0:
            raise RuntimeError("No frames extracted from video")

        print(f"[XScale] Extracted {total_frames} frames")

        if progress_callback:
            progress_callback(0.05)

        # ── Step 2: Upscale frames ──────────────────────
        print(f"[XScale] Upscaling with {model['name']} at {scale}x")

        # Use the ncnn-vulkan binary for batch processing
        subprocess.run(
            [
                model["binary"],
                "-i", frames_dir,
                "-o", upscaled_dir,
                "-n", model["model_name"],
                "-s", str(scale),
                "-f", "png",
            ],
            check=True,
            capture_output=True,
        )

        # If the binary doesn't support progress, we estimate it
        # by checking the output directory periodically
        upscaled_files = sorted(glob.glob(f"{upscaled_dir}/frame_*.png"))
        upscaled_count = len(upscaled_files)

        if progress_callback:
            progress_callback(0.75)

        print(f"[XScale] Upscaled {upscaled_count}/{total_frames} frames")

        # ── Step 3: Extract audio from original ─────────
        audio_path = "/content/temp_audio.aac"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vn", "-acodec", "aac",
                "-b:a", "192k",
                audio_path,
            ],
            capture_output=True,
        )
        has_audio = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0

        # ── Step 4: Get original FPS ────────────────────
        fps_result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_path,
            ],
            capture_output=True,
            text=True,
        )
        fps = fps_result.stdout.strip() or "30"

        if progress_callback:
            progress_callback(0.85)

        # ── Step 5: Reassemble video (H.265 / HEVC) ────
        print(f"[XScale] Reassembling at {fps} FPS with H.265")

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", fps,
            "-i", f"{upscaled_dir}/frame_%06d.png",
        ]

        if has_audio:
            ffmpeg_cmd.extend(["-i", audio_path])

        ffmpeg_cmd.extend([
            "-c:v", "libx265",
            "-preset", "medium",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
        ])

        if has_audio:
            ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k"])

        ffmpeg_cmd.append(output_path)

        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

        if progress_callback:
            progress_callback(1.0)

        print(f"[XScale] ✅ Output saved to {output_path}")

    finally:
        # Cleanup temporary frame directories
        _cleanup_dir(frames_dir)
        _cleanup_dir(upscaled_dir)
        if os.path.exists("/content/temp_audio.aac"):
            os.remove("/content/temp_audio.aac")


def _cleanup_dir(path: str):
    """Remove all files in a directory."""
    if os.path.exists(path):
        for f in glob.glob(f"{path}/*"):
            os.remove(f)
        os.rmdir(path)
