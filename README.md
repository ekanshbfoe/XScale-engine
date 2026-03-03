# XScale Engine 🚀

AI-powered video upscaling engine that runs on Google Colab. Used by the [XScale mobile app](https://github.com/YOUR_USERNAME/XScale-app) to enhance videos using Real-ESRGAN and Real-CUGAN.

## Quick Start

1. **Open the notebook** → [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/XScale-engine/blob/main/XScale_Engine.ipynb)

2. **Add your Firebase credentials** to Colab Secrets (🔑 icon in sidebar):
   - Secret name: `FIREBASE_SERVICE_ACCOUNT`
   - Value: paste the entire JSON content of your Firebase service account key

3. **Click "Runtime → Run all"** — the engine starts automatically!

## Architecture

```
Mobile App → Upload to Drive → POST /upscale → Colab Engine
                                                    ↓
                                            Download from Drive
                                            AI Upscaling (ESRGAN/CUGAN)
                                            Upload result to Drive
                                                    ↓
                                            Firebase progress updates → App
```

## Files

| File | Description |
|------|-------------|
| `bridge.py` | FastAPI server + Cloudflare tunnel + Firebase integration |
| `processor.py` | AI upscaling pipeline (FFmpeg + ESRGAN/CUGAN) |
| `cleanup.py` | Auto-purge Drive storage when over 5GB |
| `XScale_Engine.ipynb` | Ready-to-run Colab notebook |
| `requirements.txt` | Python dependencies |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upscale` | Start video processing |
| `GET` | `/status` | Get engine status + progress |
| `POST` | `/heartbeat` | Keep engine alive |
| `POST` | `/stop` | Graceful shutdown |

## License

MIT
