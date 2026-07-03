"""Central hub for every tool in this project.

    streamlit run hub.py

One place to run and inspect:
  - the YOLO detector+tracker (run_algo.py / app.py)
  - the camera-motion estimator (camera_motion.py)
  - the robustness studies (robustness.py, video_robustness.py, failure_sweep.py)
  - the OTB benchmark tools (run_otb.py, otb_eval.py, run_ds_tracker.py)
  - the correlation studies (camera_motion_otb.py, shape_complexity.py)
  - all generated results, and the written reports.

Every tool runs as a subprocess of its existing CLI script (same commands you
would type in a terminal), with live logs streamed into the page and the
output artifacts displayed underneath.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
SAMPLES = BASE / "samples"
UPLOADS = BASE / "uploads"
OTB_DIR = BASE / "ds" / "OTB-dataset" / "OTB_downloads"
PY = sys.executable

st.set_page_config(page_title="Video Research Hub", page_icon="🎛️", layout="wide")


# ── shared helpers ─────────────────────────────────────────────────────────────

def run_tool(args: list[str], label: str) -> bool:
    """Run a project script as a subprocess, streaming its log into the page."""
    cmd = [PY, *args]
    with st.status(f"Running {label}…", expanded=True) as status:
        st.code("python3 " + " ".join(args), language="bash")
        log_box = st.empty()
        lines: list[str] = []
        proc = subprocess.Popen(
            cmd, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())
            log_box.code("\n".join(lines[-25:]) or "…")
        proc.wait()
        ok = proc.returncode == 0
        status.update(
            label=f"{label} — {'done' if ok else f'FAILED (exit {proc.returncode})'}",
            state="complete" if ok else "error",
            expanded=not ok,
        )
    return ok


def pick_video(key: str) -> Path | None:
    """Choose a video from samples/ + uploads/, or upload a new one."""
    UPLOADS.mkdir(exist_ok=True)
    vids = sorted(SAMPLES.glob("*.mp4")) + sorted(UPLOADS.glob("*"))
    vids = [v for v in vids if v.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    uploaded = st.file_uploader("Upload a video (saved to uploads/)",
                                type=["mp4", "mov", "avi", "mkv"], key=key + "_up")
    if uploaded is not None:
        dest = UPLOADS / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        st.success(f"Saved to {dest.relative_to(BASE)}")
        if dest not in vids:
            vids.append(dest)
    if not vids:
        st.warning("No videos available — upload one above.")
        return None
    labels = [str(v.relative_to(BASE)) for v in vids]
    default = labels.index(str((UPLOADS / uploaded.name).relative_to(BASE))) if uploaded else 0
    choice = st.selectbox("Video", labels, index=default, key=key + "_sel")
    return BASE / choice


FFMPEG = shutil.which("ffmpeg")


@st.cache_data(show_spinner="Re-encoding video for playback…")
def _playable(path_str: str, mtime: float) -> str:
    """OpenCV writes mp4v, which browsers can't play. When ffmpeg is available,
    transcode once to H.264 (cached by path+mtime) and serve that."""
    path = Path(path_str)
    cache = BASE / ".h264_cache"
    cache.mkdir(exist_ok=True)
    out = cache / (hashlib.md5(f"{path_str}-{mtime}".encode()).hexdigest() + ".mp4")
    if not out.exists():
        r = subprocess.run([FFMPEG, "-y", "-i", str(path), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(out)], capture_output=True)
        if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            return path_str
    return str(out)


def show_video(path: Path):
    """Inline playback (H.264-transcoded when ffmpeg exists) + download."""
    play = _playable(str(path), path.stat().st_mtime) if FFMPEG else str(path)
    st.video(play)
    with open(path, "rb") as f:
        st.download_button(f"Download {path.name}", f, file_name=path.name,
                           mime="video/mp4", key=f"dl_{path}")
    if not FFMPEG:
        st.caption("If the player above is blank the file is mp4v-encoded "
                   "(no ffmpeg on this machine to re-encode H.264) — use the download button.")


def show_artifacts(out_dir: Path, patterns: list[str] | None = None,
                   key: str = "", newest_first: bool = True):
    """Render everything in an output directory: PNGs inline, CSVs as tables,
    MP4s as players + download buttons."""
    if not out_dir.exists():
        st.info(f"`{out_dir.relative_to(BASE)}/` doesn't exist yet — run the tool above.")
        return
    files = []
    for pat in (patterns or ["*"]):
        files.extend(out_dir.glob(pat))
    files = sorted(set(f for f in files if f.is_file() and not f.name.startswith(".")),
                   key=lambda f: f.stat().st_mtime, reverse=newest_first)
    if not files:
        st.info(f"No artifacts in `{out_dir.relative_to(BASE)}/` yet.")
        return
    images = [f for f in files if f.suffix.lower() == ".png"]
    csvs = [f for f in files if f.suffix.lower() == ".csv"]
    videos = [f for f in files if f.suffix.lower() == ".mp4"]
    xlsxs = [f for f in files if f.suffix.lower() == ".xlsx"]
    mds = [f for f in files if f.suffix.lower() in {".md", ".txt"}]
    others = [f for f in files if f not in images + csvs + videos + xlsxs + mds]

    for f in images:
        st.image(str(f), caption=str(f.relative_to(BASE)), width="stretch")
    for f in mds:
        with st.expander(f"📄 {f.relative_to(BASE)}", expanded=(f.suffix == ".md")):
            if f.suffix == ".md":
                st.markdown(f.read_text())
            else:
                st.text(f.read_text())
    for f in csvs:
        st.markdown(f"**{f.relative_to(BASE)}**")
        try:
            st.dataframe(pd.read_csv(f), height=280)
        except Exception as e:
            st.warning(f"Could not read CSV: {e}")
    for f in xlsxs:
        st.markdown(f"**{f.relative_to(BASE)}** (Excel report)")
        try:
            sheets = pd.read_excel(f, sheet_name=None)
            tabs = st.tabs(list(sheets))
            for tab, (sname, df) in zip(tabs, sheets.items()):
                with tab:
                    st.dataframe(df, height=260)
        except Exception as e:
            st.warning(f"Could not read XLSX: {e}")
        with open(f, "rb") as fh:
            st.download_button(f"Download {f.name}", fh, file_name=f.name,
                               key=f"dlx_{key}_{f}")
    for f in videos:
        st.markdown(f"**{f.relative_to(BASE)}**")
        show_video(f)
    for f in others:
        with open(f, "rb") as fh:
            st.download_button(f"Download {f.name}", fh, file_name=f.name,
                               key=f"dl_{key}_{f}")


def otb_sequences() -> list[str]:
    if not OTB_DIR.exists():
        return []
    return sorted(d.name for d in OTB_DIR.iterdir()
                  if d.is_dir() and (d / "img").exists())


def model_picker(key: str, default: str = "yolo11n.pt") -> str:
    local = sorted(p.name for p in BASE.glob("yolo*.pt"))
    opts = local + [m for m in ["yolo11s.pt", "yolo11m.pt"] if m not in local]
    return st.selectbox("Model", opts, index=opts.index(default) if default in opts else 0,
                        key=key, help="Models not on disk are auto-downloaded by Ultralytics.")


# ── pages ──────────────────────────────────────────────────────────────────────

def page_dashboard():
    st.title("🎛️ Video Research Hub")
    st.markdown(
        "One place to run every tool in this project and browse its results. "
        "Pick a tool from the sidebar; each page runs the underlying CLI script "
        "with live logs and shows the artifacts it produces."
    )

    c1, c2, c3 = st.columns(3)
    seqs = otb_sequences()
    c1.metric("OTB sequences on disk", len(seqs))
    n_videos = len(list(SAMPLES.glob('*.mp4'))) if SAMPLES.exists() else 0
    c2.metric("Sample videos", n_videos)
    out_dirs = [d for d in ["run_out", "robustness_out", "video_robustness_out",
                            "camera_motion_out", "otb_eval_out", "otb_runs",
                            "results", "complexity_out", "failure_sweep_out",
                            "brightness_out", "model_compare_out", "case_analysis_out",
                            "telemetry_out", "tracker_anatomy_out"]
                if (BASE / d).exists()]
    c3.metric("Result folders", len(out_dirs))

    st.subheader("Key findings so far")
    st.markdown("""
- **Blur is catastrophic, lighting is benign** — motion/Gaussian blur erase 83–87 %
  of mAP@0.5 at high severity; brightness across 0.2×–3× costs at most 30 %.
- **It's a cliff, not a slope** — fine up to ~5 px motion blur / σ ≈ 2, then collapse.
- **Object size dominates blur robustness** — 25 px blur keeps 59 % of large objects
  but only 14 % of small/distant ones.
- **Failure mode is missed objects, not false alarms** — recall collapses, precision holds.

Full write-up on the **Reports** page.
""")

    if (BASE / "results" / "summary.csv").exists():
        st.subheader("CSRT tracker summary (OTB)")
        st.dataframe(pd.read_csv(BASE / "results" / "summary.csv"), height=250)


def page_tracker():
    st.title("🎯 Detector + Tracker")
    st.markdown("Run the YOLO11 detection + tracking pipeline (`run_algo.py`) on any video. "
                "The full interactive app is still available separately: `streamlit run app.py`.")
    video = pick_video("trk")
    c1, c2, c3 = st.columns(3)
    with c1:
        model = model_picker("trk_model")
    with c2:
        tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"], key="trk_tracker")
    with c3:
        conf = st.slider("Confidence", 0.05, 0.95, 0.30, 0.05, key="trk_conf")
    single = st.checkbox("Track a single object only (largest first detection, re-locks on loss)",
                         value=True, key="trk_single")

    if video and st.button("▶ Run tracker", type="primary"):
        args = ["run_algo.py", str(video), "--model", model,
                "--tracker", tracker, "--conf", str(conf)]
        if single:
            args += ["--single", "auto"]
        run_tool(args, "run_algo.py")
    st.divider()
    st.subheader("Outputs (run_out/)")
    show_artifacts(BASE / "run_out", key="trk")


def page_camera_motion():
    st.title("🎥 Camera Motion Estimator")
    st.markdown("Estimates per-frame camera translation / rotation / zoom from optical flow "
                "with RANSAC background separation (`camera_motion.py`).")
    video = pick_video("cam")
    c1, c2, c3 = st.columns(3)
    with c1:
        save_video = st.checkbox("Save overlay video (slower)", key="cam_sv")
    with c2:
        horizon = st.number_input("Horizon y (px, 0 = off)", 0, 4000, 0, key="cam_hy")
    with c3:
        pitch = st.number_input("Camera pitch (deg, 0 = off)", 0.0, 89.0, 0.0, key="cam_pd")

    if video and st.button("▶ Estimate camera motion", type="primary"):
        args = ["camera_motion.py", str(video), "--out-dir", "camera_motion_run"]
        if save_video:
            args.append("--save-video")
        if horizon > 0:
            args += ["--horizon-y", str(int(horizon))]
        if pitch > 0:
            args += ["--pitch-deg", str(pitch)]
        run_tool(args, "camera_motion.py")
    st.divider()
    st.subheader("Outputs (camera_motion_run/)")
    show_artifacts(BASE / "camera_motion_run", key="cam")


def page_robustness():
    st.title("🧪 Robustness — still images (COCO128)")
    st.markdown("Sweeps motion blur, Gaussian blur and brightness over COCO128 and measures "
                "mAP/recall with Ultralytics validation against real ground truth "
                "(`robustness.py`). Takes a while; downloads COCO128 on first run.")
    if st.button("▶ Run still-image study", type="primary"):
        run_tool(["robustness.py"], "robustness.py")
    st.divider()
    st.subheader("Outputs (robustness_out/)")
    show_artifacts(BASE / "robustness_out", key="rob")


def page_video_robustness():
    st.title("📼 Robustness — real video")
    st.markdown("Applies rising blur / lighting changes to a real clip and measures how many "
                "of the clean video's detections survive (`video_robustness.py`).")
    video = pick_video("vrob")
    make_videos = st.checkbox("Render worst-case annotated clips (--make-videos)", key="vrob_mv")
    if video and st.button("▶ Run video study", type="primary"):
        args = ["video_robustness.py", str(video)]
        if make_videos:
            args.append("--make-videos")
        run_tool(args, "video_robustness.py")
    st.divider()
    st.subheader("Outputs (video_robustness_out/)")
    show_artifacts(BASE / "video_robustness_out", key="vrob")


def page_otb():
    st.title("🏀 OTB Benchmark")
    seqs = otb_sequences()
    if not seqs:
        st.error("No OTB sequences found under `ds/OTB-dataset/OTB_downloads/`. "
                 "Run `ds/OTB-dataset/download.py` first.")
        return

    tab_run, tab_eval, tab_csrt, tab_sweep = st.tabs(
        ["Visualize one sequence", "YOLO evaluation", "CSRT baseline", "Failure sweep"])

    with tab_run:
        st.markdown("Run the YOLO tracker on one sequence and render an annotated MP4 "
                    "(`run_otb.py`).")
        seq = st.selectbox("Sequence", seqs, key="otb_seq")
        c1, c2, c3 = st.columns(3)
        with c1:
            model = model_picker("otb_model")
        with c2:
            conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05, key="otb_conf")
        with c3:
            show_gt = st.checkbox("Overlay ground truth", value=True, key="otb_gt")
        all_objects = st.checkbox("Annotate ALL objects (default follows only the GT target)",
                                  value=False, key="otb_all")
        if st.button("▶ Run on sequence", type="primary", key="otb_run"):
            args = ["run_otb.py", seq, "--model", model, "--conf", str(conf)]
            if show_gt:
                args.append("--show-gt")
            if all_objects:
                args.append("--all-objects")
            run_tool(args, "run_otb.py")
        st.subheader("Outputs (otb_runs/)")
        show_artifacts(BASE / "otb_runs", key="otbrun")

    with tab_eval:
        st.markdown("Proper OTB scoring of YOLO tracking — success AUC, precision@20px, "
                    "mean IoU per sequence (`otb_eval.py`).")
        pick = st.multiselect("Sequences (empty = all)", seqs, key="oe_seqs")
        model = model_picker("oe_model")
        if st.button("▶ Evaluate", type="primary", key="oe_run"):
            args = ["otb_eval.py", "--model", model]
            if pick:
                args += ["--seqs", *pick]
            run_tool(args, "otb_eval.py")
        st.subheader("Outputs (otb_eval_out/)")
        show_artifacts(BASE / "otb_eval_out", key="oe")

    with tab_csrt:
        st.markdown("Classical single-object baseline: OpenCV CSRT initialized from the "
                    "first-frame GT box, run on **every** sequence (`run_ds_tracker.py`). "
                    "The correlation studies depend on its `results/` output.")
        if st.button("▶ Run CSRT on all sequences", type="primary", key="csrt_run"):
            run_tool(["run_ds_tracker.py"], "run_ds_tracker.py")
        st.subheader("Summary (results/summary.csv)")
        show_artifacts(BASE / "results", patterns=["summary.csv"], key="csrt")

    with tab_sweep:
        st.markdown("Systematic failure sweep: every corruption × severity × 20 confidence "
                    "thresholds against clean-pass pseudo-GT (`failure_sweep.py`). Slow.")
        pick = st.multiselect("Sequences (empty = all)", seqs, key="fs_seqs")
        stride = st.slider("Frame stride (higher = faster)", 1, 10, 3, key="fs_stride")
        if st.button("▶ Run sweep", type="primary", key="fs_run"):
            args = ["failure_sweep.py", "--stride", str(stride)]
            if pick:
                args += ["--seqs", *pick]
            run_tool(args, "failure_sweep.py")
        st.subheader("Outputs (failure_sweep_out/)")
        show_artifacts(BASE / "failure_sweep_out", key="fs")


def page_correlations():
    st.title("🔬 Correlation Studies")
    st.markdown("Both studies join per-sequence metrics with CSRT tracking IoU from "
                "`results/` — run the **CSRT baseline** (OTB page) first if it's empty.")

    tab_cam, tab_shape = st.tabs(["Camera motion vs IoU", "Shape complexity vs IoU"])

    with tab_cam:
        st.markdown("Does more camera motion hurt tracking? Optical-flow camera-motion "
                    "metrics vs mean IoU, Spearman ρ (`camera_motion_otb.py`).")
        if st.button("▶ Run camera-motion correlation", type="primary", key="cmo_run"):
            run_tool(["camera_motion_otb.py"], "camera_motion_otb.py")
        st.subheader("Outputs (camera_motion_out/)")
        show_artifacts(BASE / "camera_motion_out", key="cmo")

    with tab_shape:
        st.markdown("Does target visual complexity (silhouette, texture, contrast…) predict "
                    "tracking quality? (`shape_complexity.py`)")
        if st.button("▶ Run shape-complexity study", type="primary", key="shp_run"):
            run_tool(["shape_complexity.py"], "shape_complexity.py")
        st.subheader("Outputs (complexity_out/)")
        show_artifacts(BASE / "complexity_out", key="shp")


def page_how_tracker_works():
    st.title("🧠 How the Tracker Works")
    tab_doc, tab_demo = st.tabs(["The explanation", "See it live (tracker anatomy)"])
    with tab_doc:
        doc = BASE / "docs" / "HOW_THE_TRACKER_WORKS.md"
        if doc.exists():
            st.markdown(doc.read_text())
        else:
            st.error("docs/HOW_THE_TRACKER_WORKS.md is missing.")
    with tab_demo:
        st.markdown("Runs the **real BYTETracker** next to YOLO and renders its internals: "
                    "green = raw detections, yellow corners = Kalman-predicted boxes, "
                    "solid = tracks with IDs, plus birth/death flashes (`tracker_anatomy.py`).")
        mode = st.radio("Source", ["Video file", "OTB sequence"], horizontal=True, key="ta_mode")
        seq = None
        video = None
        if mode == "OTB sequence":
            seqs = otb_sequences()
            seq = st.selectbox("Sequence", seqs, key="ta_seq") if seqs else None
        else:
            video = pick_video("ta")
        c1, c2 = st.columns(2)
        with c1:
            conf = st.slider("Detector confidence", 0.05, 0.9, 0.25, 0.05, key="ta_conf",
                             help="Set to 0.10 to let ByteTrack's low-confidence second "
                                  "association fire (see the doc, §5).")
        with c2:
            maxf = st.number_input("Max frames", 30, 2000, 150, 30, key="ta_maxf")
        if st.button("▶ Run tracker anatomy", type="primary", key="ta_run"):
            args = ["tracker_anatomy.py", "--conf", str(conf), "--max-frames", str(int(maxf))]
            if seq:
                args += ["--seq", seq]
            elif video:
                args.insert(1, str(video))
            run_tool(args, "tracker_anatomy.py")
        st.subheader("Outputs (tracker_anatomy_out/)")
        show_artifacts(BASE / "tracker_anatomy_out", key="ta")


def page_brightness():
    st.title("💡 Brightness Meter")
    st.markdown("Per-frame luma, contrast and clipping analysis with an exposure classification "
                "and a detection-risk verdict wired to the robustness findings "
                "(`brightness_meter.py`). Pick two videos for an A/B overlay.")
    video = pick_video("br")
    compare = st.selectbox(
        "Compare against (optional)",
        ["(none)"] + [str(p.relative_to(BASE)) for p in
                      sorted((BASE / "video_robustness_out").glob("*.mp4")) +
                      sorted(SAMPLES.glob("*.mp4")) if p.exists()],
        key="br_cmp")
    c1, c2 = st.columns(2)
    with c1:
        maxf = st.number_input("Max frames (0 = all)", 0, 5000, 0, 100, key="br_maxf")
    with c2:
        stride = st.slider("Frame stride", 1, 10, 2, key="br_stride")
    if video and st.button("▶ Measure brightness", type="primary", key="br_run"):
        args = ["brightness_meter.py", str(video)]
        if compare != "(none)":
            args.append(str(BASE / compare))
        args += ["--stride", str(stride)]
        if maxf > 0:
            args += ["--max-frames", str(int(maxf))]
        run_tool(args, "brightness_meter.py")
    st.subheader("Outputs (brightness_out/)")
    show_artifacts(BASE / "brightness_out", key="br")


def page_model_zoo():
    st.title("🧬 Model Zoo — other datasets")
    st.markdown("Compares models **trained on different datasets** on the same clip: "
                "yolo11n/s (COCO), yolov8s-oiv7 (Open Images V7, 601 classes), "
                "yolov8s-worldv2 (open-vocabulary). Reports det/frame, confidence, fps, "
                "cross-model box agreement, and a side-by-side annotated frame "
                "(`model_compare.py`).")
    video = pick_video("mz")
    all_models = ["yolo11n.pt", "yolo11s.pt", "yolov8s-oiv7.pt", "yolov8s-worldv2.pt",
                  "yolo11l.pt", "rtdetr-l.pt"]
    models = st.multiselect("Models", all_models, default=all_models[:4], key="mz_models")
    c1, c2, c3 = st.columns(3)
    with c1:
        maxf = st.number_input("Max frames", 20, 1000, 100, 20, key="mz_maxf")
    with c2:
        stride = st.slider("Stride", 1, 10, 2, key="mz_stride")
    with c3:
        conf = st.slider("Confidence", 0.05, 0.9, 0.3, 0.05, key="mz_conf")
    if video and models and st.button("▶ Compare models", type="primary", key="mz_run"):
        run_tool(["model_compare.py", str(video), "--models", *models,
                  "--max-frames", str(int(maxf)), "--stride", str(stride),
                  "--conf", str(conf)], "model_compare.py")
    st.subheader("Outputs (model_compare_out/)")
    show_artifacts(BASE / "model_compare_out", key="mz")


def page_cases():
    st.title("🗂️ Success / Failure Cases")
    st.markdown("Joins every per-sequence result in the project (YOLO OTB eval, CSRT baseline, "
                "catalog, camera motion, complexity, corruption sweep) and divides sequences "
                "into SUCCESS / PARTIAL / FAILURE tiers, ranks the factors that separate them, "
                "and writes the `CASES.md` report (`case_analysis.py`).")
    c1, c2 = st.columns(2)
    with c1:
        s_thr = st.slider("SUCCESS threshold (mean IoU ≥)", 0.3, 0.8, 0.5, 0.05, key="ca_s")
    with c2:
        f_thr = st.slider("FAILURE threshold (mean IoU <)", 0.1, 0.5, 0.3, 0.05, key="ca_f")
    if st.button("▶ Run case analysis", type="primary", key="ca_run"):
        run_tool(["case_analysis.py", "--success-thr", str(s_thr),
                  "--failure-thr", str(f_thr)], "case_analysis.py")
    st.subheader("Outputs (case_analysis_out/)")
    show_artifacts(BASE / "case_analysis_out", key="ca")


def page_telemetry():
    st.title("📊 Telemetry Overlay + Excel report")
    st.markdown("Burns live tracker telemetry into the output video (detections, track IDs, "
                "births/deaths, confidence, brightness — plus IoU vs ground truth and running "
                "accuracy on OTB sequences) and writes a 3-sheet Excel report: per-frame data, "
                "summary, and per-track statistics (`telemetry_overlay.py`).")
    mode = st.radio("Source", ["Video file", "OTB sequence (with ground-truth accuracy)"],
                    horizontal=True, key="te_mode")
    seq = None
    video = None
    if mode.startswith("OTB"):
        seqs = otb_sequences()
        seq = st.selectbox("Sequence", seqs, key="te_seq") if seqs else None
    else:
        video = pick_video("te")
    c1, c2, c3 = st.columns(3)
    with c1:
        model = model_picker("te_model")
    with c2:
        conf = st.slider("Confidence", 0.05, 0.9, 0.3, 0.05, key="te_conf")
    with c3:
        maxf = st.number_input("Max frames (0 = all)", 0, 5000, 300, 50, key="te_maxf")
    single = st.checkbox("Track a single target only (GT-seeded on OTB; re-locks on loss)",
                         value=True, key="te_single")
    if (seq or video) and st.button("▶ Run telemetry", type="primary", key="te_run"):
        args = ["telemetry_overlay.py", "--model", model, "--conf", str(conf)]
        if single:
            args.append("--single")
        if maxf > 0:
            args += ["--max-frames", str(int(maxf))]
        if seq:
            args += ["--seq", seq]
        else:
            args.insert(1, str(video))
        run_tool(args, "telemetry_overlay.py")
    st.subheader("Outputs (telemetry_out/)")
    show_artifacts(BASE / "telemetry_out", key="te")


def page_methods():
    st.title("🛠️ Methods / Toolbox")
    st.markdown("Every tool in this project: what it does, how it works, and the exact "
                "command to recreate its results yourself. All scripts live in the project "
                "root and run with `python3` from there.")
    tools = [
        ("Detector + Tracker", "run_algo.py", "run_out/",
         "YOLO11 detection + ByteTrack/BoT-SORT tracking on any video; writes the annotated "
         "clip and sample frames. Same pipeline as the Streamlit app. `--single auto` (or an "
         "ID) follows ONE object only, re-locking via `single_target.TargetFollower` when the "
         "tracker drops the ID; `run_otb.py` does the same by default, seeded from ground truth.",
         "python3 run_algo.py samples/people.mp4 --single auto"),
        ("How the tracker works", "docs/HOW_THE_TRACKER_WORKS.md + tracker_anatomy.py",
         "tracker_anatomy_out/",
         "Source-grounded explainer of ByteTrack/BoT-SORT (Kalman prediction, two-round "
         "IoU association via lap.lapjv, track lifecycle), plus a demo that runs the real "
         "BYTETracker and renders predictions, matches, births and deaths per frame.",
         "python3 tracker_anatomy.py --seq Basketball --max-frames 200"),
        ("Brightness meter", "brightness_meter.py", "brightness_out/",
         "Per-frame BT.709 luma, contrast, shadow/highlight clipping; classifies exposure "
         "into 5 bands and issues a detection-risk verdict based on the robustness findings "
         "(over-exposure is the dangerous direction). Import `measure_brightness()` to reuse.",
         "python3 brightness_meter.py samples/people.mp4 video_robustness_out/people_dark.mp4"),
        ("Model zoo (other datasets)", "model_compare.py", "model_compare_out/",
         "Runs COCO models (yolo11n/s), an Open Images V7 model (601 classes) and YOLO-World "
         "open-vocab on the same frames; compares det/frame, confidence, fps, class-agnostic "
         "box agreement vs the yolo11n reference, and renders a side-by-side annotated frame.",
         "python3 model_compare.py samples/traffic.mp4 --max-frames 100"),
        ("Success/failure case analysis", "case_analysis.py", "case_analysis_out/",
         "Joins all per-sequence results (YOLO + CSRT scores, catalog attributes, camera "
         "motion, target complexity, corruption sweep), tiers sequences into SUCCESS/PARTIAL/"
         "FAILURE, ranks separating factors (Spearman + Mann-Whitney + depth-2 decision "
         "tree), finds corruption knees, and writes the CASES.md report.",
         "python3 case_analysis.py"),
        ("Telemetry overlay + xlsx", "telemetry_overlay.py", "telemetry_out/",
         "Tracks a video (or OTB sequence with GT), burns a live HUD into every frame "
         "(detections, IDs, births/deaths, confidence, brightness, IoU, running accuracy) "
         "and writes a 3-sheet Excel report: per_frame / summary / tracks. `--single` draws "
         "and scores only one followed target (re-locks on loss, switches counted in the xlsx).",
         "python3 telemetry_overlay.py --seq Basketball --max-frames 300 --single"),
        ("Camera motion estimator", "camera_motion.py", "camera_motion_run/",
         "Lucas-Kanade sparse optical flow + RANSAC affine on the background → per-frame "
         "camera translation/rotation/zoom; CSV + plots (+ overlay video).",
         "python3 camera_motion.py samples/people.mp4 --out-dir camera_motion_run"),
        ("Still-image robustness", "robustness.py", "robustness_out/",
         "Corrupts COCO128 (motion blur / gaussian blur / brightness, severity 0 = clean) "
         "and measures mAP/recall with Ultralytics val against real labels.",
         "python3 robustness.py"),
        ("Video robustness", "video_robustness.py", "video_robustness_out/",
         "Same corruptions on real clips; clean-video detections serve as pseudo-ground-truth; "
         "measures retention/confidence/IoU per severity.",
         "python3 video_robustness.py samples/traffic.mp4 --make-videos"),
        ("OTB failure sweep", "failure_sweep.py", "failure_sweep_out/",
         "Corruption × severity × 20 confidence thresholds over OTB sequences vs clean-pass "
         "pseudo-GT — the raw data behind the corruption knees.",
         "python3 failure_sweep.py --seqs Car1 BlurBody --stride 3"),
        ("YOLO OTB evaluation", "otb_eval.py", "otb_eval_out/",
         "Locks onto one track ID per OTB sequence and scores it as a single-target tracker "
         "(success AUC, precision@20px, mean IoU + detector-ceiling IoU).",
         "python3 otb_eval.py --seqs Basketball Bolt"),
        ("CSRT classical baseline", "run_ds_tracker.py", "results/",
         "OpenCV CSRT initialized from the first-frame GT box on every OTB sequence — the "
         "classical baseline the correlation studies join against.",
         "python3 run_ds_tracker.py"),
        ("Camera motion vs IoU", "camera_motion_otb.py", "camera_motion_out/",
         "Runs the camera-motion estimator over OTB frames and Spearman-correlates its "
         "metrics with CSRT tracking IoU.",
         "python3 camera_motion_otb.py"),
        ("Shape complexity vs IoU", "shape_complexity.py", "complexity_out/",
         "GrabCut segmentation of the target → silhouette/texture/color/contrast metrics, "
         "correlated with per-frame tracking IoU.",
         "python3 shape_complexity.py"),
    ]
    for name, script, outdir, how, cmd in tools:
        with st.expander(f"**{name}**  —  `{script}`"):
            st.markdown(how)
            st.code(cmd, language="bash")
            st.caption(f"Outputs → `{outdir}`")


def page_results():
    st.title("📁 Results Browser")
    st.markdown("Browse everything any tool has produced.")
    dirs = {
        "run_out — tracker runs": "run_out",
        "robustness_out — still-image study": "robustness_out",
        "video_robustness_out — video study": "video_robustness_out",
        "camera_motion_run — camera motion (hub runs)": "camera_motion_run",
        "camera_motion_out — camera motion vs IoU": "camera_motion_out",
        "complexity_out — shape complexity": "complexity_out",
        "otb_eval_out — YOLO OTB scores": "otb_eval_out",
        "otb_runs — annotated OTB clips": "otb_runs",
        "failure_sweep_out — failure sweep": "failure_sweep_out",
        "results — CSRT per-sequence outputs": "results",
        "verify_out — pipeline verification": "verify_out",
        "brightness_out — brightness meter": "brightness_out",
        "model_compare_out — model zoo": "model_compare_out",
        "case_analysis_out — success/failure cases": "case_analysis_out",
        "telemetry_out — telemetry + xlsx": "telemetry_out",
        "tracker_anatomy_out — tracker internals": "tracker_anatomy_out",
    }
    existing = {label: d for label, d in dirs.items() if (BASE / d).exists()}
    label = st.selectbox("Folder", list(existing))
    folder = BASE / existing[label]

    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
    if subdirs:
        sub = st.selectbox("Subfolder", ["(top level)"] + [d.name for d in subdirs])
        if sub != "(top level)":
            folder = folder / sub
    show_artifacts(folder, key="browse")


def page_reports():
    st.title("📄 Reports")
    tab_conc, tab_readme, tab_pdf = st.tabs(["CONCLUSIONS.md", "README.md", "Export PDF"])
    with tab_conc:
        st.markdown((BASE / "CONCLUSIONS.md").read_text())
    with tab_readme:
        st.markdown((BASE / "README.md").read_text())
    with tab_pdf:
        st.markdown("Re-render `CONCLUSIONS.pdf` via headless Chrome (`make_pdf.py`).")
        if st.button("▶ Rebuild PDF", type="primary"):
            run_tool(["make_pdf.py", "CONCLUSIONS.md"], "make_pdf.py")
        pdf = BASE / "CONCLUSIONS.pdf"
        if pdf.exists():
            with open(pdf, "rb") as f:
                st.download_button("Download CONCLUSIONS.pdf", f,
                                   file_name="CONCLUSIONS.pdf", mime="application/pdf")


# ── navigation ────────────────────────────────────────────────────────────────

pages = st.navigation({
    "Overview": [
        st.Page(page_dashboard, title="Dashboard", icon="🎛️", default=True),
    ],
    "Run tools": [
        st.Page(page_tracker, title="Detector + Tracker", icon="🎯"),
        st.Page(page_camera_motion, title="Camera Motion", icon="🎥"),
        st.Page(page_robustness, title="Robustness (stills)", icon="🧪"),
        st.Page(page_video_robustness, title="Robustness (video)", icon="📼"),
        st.Page(page_otb, title="OTB Benchmark", icon="🏀"),
        st.Page(page_correlations, title="Correlation Studies", icon="🔬"),
    ],
    "Research tools": [
        st.Page(page_how_tracker_works, title="How the Tracker Works", icon="🧠"),
        st.Page(page_brightness, title="Brightness Meter", icon="💡"),
        st.Page(page_model_zoo, title="Model Zoo", icon="🧬"),
        st.Page(page_cases, title="Success/Failure Cases", icon="🗂️"),
        st.Page(page_telemetry, title="Telemetry + XLSX", icon="📊"),
    ],
    "Browse": [
        st.Page(page_results, title="Results Browser", icon="📁"),
        st.Page(page_methods, title="Methods / Toolbox", icon="🛠️"),
        st.Page(page_reports, title="Reports", icon="📄"),
    ],
})
pages.run()
