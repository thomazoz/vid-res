# Deploying the hub to Streamlit Community Cloud

The repo is already prepared for the cloud: `hub.py` is the entry point,
`requirements.txt` lists every dependency, `packages.txt` installs ffmpeg
(so videos play in the browser) and OpenCV's system libraries, and
`.gitignore` keeps the big local data (OTB dataset, conda env, large weights)
out of the repo.

## One-time setup (≈5 minutes)

1. **Push to GitHub** (from this folder):

   ```bash
   # create an empty repo named e.g. "vid-res" on github.com (private is fine), then:
   git remote add origin https://github.com/<your-username>/vid-res.git
   git push -u origin main
   ```

   Or with the GitHub CLI: `gh repo create vid-res --private --source . --push`

2. **Deploy on Streamlit:**
   - Go to https://share.streamlit.io and sign in with GitHub.
   - "Create app" → "Deploy a public app from GitHub".
   - Repository: `<your-username>/vid-res`, branch `main`, main file **`hub.py`**.
   - Deploy. First build takes several minutes (PyTorch is large).

Pushing new commits to `main` redeploys automatically.

## What works in the cloud vs locally

| | Local (this Mac) | Streamlit Cloud |
|---|---|---|
| Detector + Tracker, Telemetry, Brightness, Camera Motion, Model Zoo, Tracker Anatomy | ✅ (MPS) | ✅ CPU — slower; keep "max frames" modest |
| Existing results (plots, CSVs, CASES.md, xlsx) | ✅ | ✅ committed to the repo |
| Video playback in the browser | needs `brew install ffmpeg` | ✅ (ffmpeg via packages.txt) |
| OTB pages (benchmark, failure sweep, CSRT) | ✅ | ⚠️ dataset not in repo — pages show a friendly warning |
| COCO128 robustness study | ✅ | ⚠️ downloads COCO128 at runtime; may exceed free-tier resources |

Uploads and generated outputs on the cloud are **ephemeral** — they vanish
when the app restarts. Use the download buttons to keep results.

Resource note: the free tier gives ~2.7 GB RAM and a shared CPU. yolo11n
runs fine; the bigger zoo models (yolo11s, oiv7, worldv2) download on first
use and also fit, but run them one at a time.
