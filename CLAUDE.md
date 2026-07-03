# Project: Video Object Tracker — robustness & tracking-quality study

## What this project is

A YOLO11 video object detection + tracking app (Streamlit) plus a series of
experiments quantifying **what makes detection/tracking fail**: motion blur,
defocus blur, lighting, camera motion, and target shape complexity. Evaluated on
COCO128 (labeled stills), real sample clips, and the OTB single-object-tracking
benchmark.

- **Environment:** macOS (Apple M4 Pro, MPS), Python via local conda env in
  `.condaenv/`. Now a git repo (big data gitignored) prepared for Streamlit
  Community Cloud — see `DEPLOY.md`; entry point `hub.py`. No `ffmpeg` on this
  machine (the hub/app warn and fall back to download buttons; the cloud gets
  ffmpeg via `packages.txt`).
- **Models:** `yolo11n.pt` (default, all reported numbers) and `yolo11l.pt`
  checked into the project root. Trackers: ByteTrack (default) / BoT-SORT —
  both need the `lap` package (already in `requirements.txt`).
- **Run things with `python3`** from the project root; scripts use relative
  paths like `ds/...` and `results/...` (except `run_ds_tracker.py`, which
  resolves paths relative to itself).

## The central hub

`hub.py` — **the entry point for everything** (`streamlit run hub.py`, or
double-click `Open Hub.command` in Finder; use the
anaconda python at `/opt/anaconda3`, not `.condaenv` which lacks streamlit).
A multi-page Streamlit app (st.navigation) that runs every project script as a
subprocess with live logs and displays its artifacts: Dashboard, Detector +
Tracker, Camera Motion (writes to `camera_motion_run/`), Robustness (stills +
video), OTB Benchmark (visualize / YOLO eval / CSRT baseline / failure sweep),
Correlation Studies, Results Browser, Reports. Uploads land in `uploads/`.
A "Research tools" nav group adds: How the Tracker Works (renders
`docs/HOW_THE_TRACKER_WORKS.md` + runs `tracker_anatomy.py`), Brightness Meter,
Model Zoo, Success/Failure Cases, Telemetry + XLSX, and a Methods/Toolbox page
documenting every tool with its exact reproduce command.

## The app

`app.py` — Streamlit UI: upload a video → YOLO11 track → annotated MP4 +
plotly stats. Has custom dark CSS. Also exposes the corruption controls from
`corruptions.py`. Hardened during review: input validation (capture/dims/
writer/FPS), resource cleanup in `finally`, refuses to "succeed" on 0 frames,
warns when ffmpeg is missing (OpenCV's `mp4v` output doesn't play in browsers).

```bash
pip install -r requirements.txt
streamlit run app.py
```

`run_algo.py` — same pipeline as a headless CLI:
`python3 run_algo.py samples/people.mp4 [--model yolo11s.pt --conf 0.3 --tracker bytetrack.yaml]`
→ `run_out/`.

## Scripts (what does what)

| Script | Purpose | Output dir |
|---|---|---|
| `corruptions.py` | Library: `motion_blur`, `gaussian_blur`, `brightness` (severity 0 = identity). `SWEEPS` dict defines severity ladders. | — |
| `verify_pipeline.py` | End-to-end smoke test: synthesizes a moving clip, runs the app's exact track→write→encode loop, PASS/FAIL verdict. | `verify_out/` |
| `robustness.py` | Still-image study: corrupt COCO128, run Ultralytics `val`, get mAP/recall/precision per severity. Ground truth = real COCO labels. | `robustness_out/` |
| `video_robustness.py` | Same idea on real clips (`samples/people.mp4`, `samples/traffic.mp4`). No labels, so clean-video detections = pseudo-GT; measures retention/confidence/IoU. `--make-videos` renders worst-case clips. | `video_robustness_out/` |
| `failure_sweep.py` | Systematic sweep on OTB sequences: each corruption × severity × 20 confidence thresholds, pseudo-GT from clean pass. | `failure_sweep_out/sweep_results.csv` |
| `run_ds_tracker.py` | Runs OpenCV **CSRT** single-object tracker on every OTB sequence (init from first-frame GT box). Written when ultralytics couldn't be installed in a sandbox; kept as the classical-tracker baseline. | `results/` (per-seq dirs + `summary.csv`) |
| `run_otb.py` | Run the YOLO tracker on one OTB sequence → annotated MP4 (`--show-gt` overlays GT). | `otb_runs/` |
| `otb_eval.py` | Proper OTB evaluation of YOLO tracking: lock onto one track ID, score it as a single-target tracker (OTB success AUC, precision@20px, mean IoU, plus a detector-ceiling IoU). | `otb_eval_out/` |
| `camera_motion.py` | Estimate camera motion from any video: LK sparse optical flow + RANSAC affine → tx/ty, rotation, zoom per frame; CSV + plots. | `--out-dir` |
| `camera_motion_otb.py` | Reuses that estimator over OTB frames; Spearman-correlates camera-motion metrics with CSRT mean IoU from `results/`. | `camera_motion_out/` |
| `shape_complexity.py` | GrabCut-based target-complexity metrics (silhouette, hull ratio, entropies, edge density, FG/BG contrast) correlated with per-frame IoU. | `complexity_out/` |
| `make_pdf.py` | Markdown → styled HTML → PDF via headless Chrome (`python3 make_pdf.py CONCLUSIONS.md`). | `.pdf` next to input |
| `brightness_meter.py` | Per-frame BT.709 luma/contrast/clipping, 5-band exposure classification + detection-risk verdict; A/B overlay with 2 inputs; exposes `measure_brightness()` for reuse. | `brightness_out/` |
| `tracker_anatomy.py` | Runs a real BYTETracker next to YOLO and renders internals: Kalman-predicted boxes, matches, births/deaths + timeline. `--conf 0.1` enables the low-conf 2nd association. Doc: `docs/HOW_THE_TRACKER_WORKS.md`. | `tracker_anatomy_out/` |
| `model_compare.py` | Same clip through models trained on different datasets (yolo11n/s COCO, yolov8s-oiv7 Open Images 601-class, yolov8s-worldv2 open-vocab, `--full` adds rtdetr-l); agreement vs yolo11n reference + side-by-side frame. | `model_compare_out/` |
| `case_analysis.py` | Joins ALL per-sequence tables → SUCCESS/PARTIAL/FAILURE tiers per tracker, factor ranking (Spearman/Mann-Whitney/decision tree), corruption knees, `CASES.md` report. | `case_analysis_out/` |
| `telemetry_overlay.py` | Burns a live HUD into the video (dets, IDs, births/deaths, conf, brightness; + IoU & running accuracy with `--seq <OTB>`) and writes a 3-sheet xlsx (per_frame/summary/tracks). `--single` follows one target only. | `telemetry_out/` |
| `single_target.py` | Library: `TargetFollower` — lock onto ONE object in multi-object tracker output, re-lock by overlap when the ID dies, count switches. Used by `run_otb.py` (default), `run_algo.py --single`, `telemetry_overlay.py --single`. | — |

## Data

- `datasets/coco128/` — COCO128 with labels (auto-downloaded by `robustness.py`).
- `samples/people.mp4` (large/close objects), `samples/traffic.mp4`
  (small/distant) — the two real test clips.
- `ds/OTB-dataset/OTB_downloads/<Seq>/` — OTB sequences (`img/*.jpg` +
  `groundtruth_rect.txt`, boxes are `x,y,w,h`). Downloaded via
  `ds/OTB-dataset/download*.py`.
- `dataset_catalog.csv` — hand-made catalog of the OTB sequences: object type,
  primary/secondary challenge, frame count, difficulty, notes.

## Key results (all yolo11n unless noted — don't re-derive these)

- **Pipeline verified working:** 72/72 frames, ~3.6 det/frame, 14 stable IDs,
  27–32 fps on MPS. Real clips track correctly.
- **Blur is catastrophic, lighting is benign.** Motion/Gaussian blur erase
  83–87% of mAP@0.5 at high severity (clean baseline 0.671 on COCO128);
  brightness across 0.2×–3× costs at most 30%.
- **Cliff, not slope:** fine up to ~5 px motion blur / sigma ≈ 2, then collapse.
- **Failure mode = missed objects, not false alarms:** recall collapses while
  precision holds (~0.45–0.5); surviving boxes stay accurate (IoU ≈ 0.85).
- **Object size dominates blur robustness:** 25 px blur retains 59% of large
  objects (people.mp4) but only 14% of small ones (traffic.mp4).
- **Brightness asymmetric:** over-exposure (3×, −30%) worse than
  under-exposure (0.2×, −11%).
- Full write-up: `CONCLUSIONS.md` (+ `.pdf`/`.html` renders via `make_pdf.py`);
  method details in `README.md`.

## Conventions & gotchas

- Severity 0 of every corruption is the identity → sweeps always include a
  free clean baseline; keep that invariant when adding corruptions.
- Pseudo-GT studies (`video_robustness.py`, `failure_sweep.py`) measure
  *degradation from clean*, not absolute accuracy — don't compare their
  numbers to `otb_eval.py` or `robustness.py`.
- `results/` = **CSRT** tracker outputs; `otb_eval_out` / `otb_runs` = **YOLO**
  tracker. `camera_motion_otb.py` and `shape_complexity.py` depend on
  `results/<seq>/predictions.csv` existing (run `run_ds_tracker.py` first).
- Matplotlib scripts use the `Agg` backend (headless-safe).
- `test_write.txt`, `bus.jpg` are scratch/test artifacts; `.condaenv/` is the
  local environment — leave them alone in searches/cleanups.
- `otb_eval_out/otb_results.csv` was regenerated 2026-07-02 with much lower
  YOLO single-target scores than the run recorded in "Key results" (e.g.
  Basketball mean IoU 0.009 vs the old 0.42) — the followed track ID dies
  within ~10 frames and never re-locks (independently replicated). Treat the
  current CSV as truth; `case_analysis.py` outputs are built on it.
- With detector `conf` > `track_low_thresh` (0.1), ByteTrack's low-confidence
  second association never fires — most project scripts run conf 0.25–0.3, so
  effectively single-stage matching. See `docs/HOW_THE_TRACKER_WORKS.md` §5.
