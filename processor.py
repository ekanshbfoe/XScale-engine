"""
XScale Engine — AI Video Processor
════════════════════════════════════
Handles the actual AI upscaling pipeline:
  1. Extract frames with FFmpeg
  2. Upscale each frame with Real-ESRGAN (PyTorch + CUDA)
  3. Reassemble into H.265 video with FFmpeg

Uses the Python `realesrgan` package for GPU-accelerated upscaling
via PyTorch + CUDA — works on every Colab GPU instance.
"""

import os
import glob
import subprocess
import cv2
import numpy as np
import torch
from typing import Callable, Optional

# ─── Model configs ─────────────────────────────────────────

MODELS = {
    "realistic": {
        "name": "RealESRGAN_x4plus",
        "scale": 4,
    },
    "anime": {
        "name": "RealESRGAN_x4plus_anime_6B",
        "scale": 4,
    },
}


def _create_upsampler(model_type: str, scale_factor: int):
    """Create a RealESRGAN upsampler with the specified model."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model_config = MODELS.get(model_type, MODELS["realistic"])

    if model_type == "anime":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=6, num_grow_ch=32, scale=4,
        )
    else:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )

    upsampler = RealESRGANer(
        scale=model_config["scale"],
        model_path=None,  # Auto-download from hub
        model=model,
        tile=512,         # Process in tiles to save VRAM
        tile_pad=10,
        pre_pad=0,
        half=True,        # FP16 for faster processing on T4
        gpu_id=0 if torch.cuda.is_available() else None,
    )

    return upsampler, model_config


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
        scale_factor: Upscale factor (2 or 4)
        model_type: "realistic" or "anime"
        progress_callback: Called with 0.0–1.0 as progress
    """
    scale = max(2, min(4, int(scale_factor)))

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

        frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
        total_frames = len(frame_files)

        if total_frames == 0:
            raise RuntimeError("No frames extracted from video")

        print(f"[XScale] Extracted {total_frames} frames")

        if progress_callback:
            progress_callback(0.05)

        # ── Step 2: Create upsampler ─────────────────────
        print(f"[XScale] Loading {model_type} model (CUDA: {torch.cuda.is_available()})")
        upsampler, model_config = _create_upsampler(model_type, scale)

        # ── Step 3: Upscale frames one by one ────────────
        print(f"[XScale] Upscaling {total_frames} frames at {scale}x")

        for idx, frame_path in enumerate(frame_files):
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=scale)

            out_path = os.path.join(
                upscaled_dir, os.path.basename(frame_path)
            )
            cv2.imwrite(out_path, output)

            # Report progress (5%–75% range for upscaling)
            if progress_callback and (idx % 5 == 0 or idx == total_frames - 1):
                p = 0.05 + (idx / total_frames) * 0.70
                progress_callback(p)

            if idx % 20 == 0:
                print(f"[XScale] Frame {idx + 1}/{total_frames}")

        if progress_callback:
            progress_callback(0.75)

        print(f"[XScale] Upscaled all {total_frames} frames")

        # ── Step 4: Extract audio from original ─────────
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

        # ── Step 5: Get original FPS ────────────────────
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

        # ── Step 6: Reassemble video (H.265 / HEVC) ────
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
