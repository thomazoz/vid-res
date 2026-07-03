import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import plotly.express as px
import streamlit as st
from ultralytics import YOLO

from corruptions import brightness, gaussian_blur, motion_blur

st.set_page_config(
    page_title="Video Object Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Overall background */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* Hide default streamlit header padding */
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 0 !important; }

/* Sidebar title */
.sidebar-title {
    font-size: 1.3rem; font-weight: 700; color: #e6edf3;
    margin-bottom: 0.2rem; letter-spacing: -0.3px;
}
.sidebar-sub {
    font-size: 0.75rem; color: #8b949e; margin-bottom: 1.2rem;
}

/* Section labels */
.section-label {
    font-size: 0.7rem; font-weight: 600; color: #58a6ff;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 1rem 0 0.4rem 0;
}

/* Main panel heading */
.panel-heading {
    font-size: 0.85rem; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 0.5rem;
}

/* Status pill */
.status-none {
    display: inline-block; padding: 2px 10px;
    background: #21262d; border: 1px solid #30363d;
    border-radius: 20px; font-size: 0.72rem; color: #8b949e;
}
.status-locked {
    display: inline-block; padding: 2px 10px;
    background: #0d2b1d; border: 1px solid #238636;
    border-radius: 20px; font-size: 0.72rem; color: #3fb950;
}

/* Run button */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #238636; border: 1px solid #2ea043;
    color: #fff; font-weight: 600; width: 100%;
    padding: 0.55rem; border-radius: 6px; font-size: 0.95rem;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #2ea043;
}

/* Slider labels */
label[data-testid="stWidgetLabel"] p { color: #c9d1d9 !important; font-size: 0.8rem !important; }

/* Select boxes */
[data-testid="stSelectbox"] label p { color: #c9d1d9 !important; font-size: 0.8rem !important; }

/* Toggle */
[data-testid="stToggle"] label p { color: #c9d1d9 !important; font-size: 0.82rem !important; }

/* File uploader */
[data-testid="stFileUploader"] label p { color: #c9d1d9 !important; font-size: 0.82rem !important; }
[data-testid="stFileUploaderDropzone"] {
    background: #21262d !important; border: 1px dashed #30363d !important;
    border-radius: 6px !important;
}

/* Captions */
[data-testid="stCaptionContainer"] p { color: #8b949e !important; font-size: 0.75rem !important; }

/* Success / warning / info boxes */
[data-testid="stAlert"] { border-radius: 6px !important; }

/* Video player */
video { border-radius: 8px; }

/* Divider */
hr { border-color: #30363d !important; margin: 0.8rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────

ACQUIRE_WINDOW = 15  # frames to wait for the boxed target to be detected + tracked


def apply_distortion(frame, bright, mblur, gblur):
    frame = brightness(frame, bright)
    frame = motion_blur(frame, mblur)
    frame = gaussian_blur(frame, gblur)
    return frame


@st.cache_data(show_spinner=False)
def get_first_frame(data: bytes):
    # Decode the first frame via a throwaway temp file, then delete it so
    # preview files don't accumulate in $TMPDIR across uploads.
    with tempfile.NamedTemporaryFile(suffix=".video", delete=False) as tf:
        tf.write(data)
        tmp = Path(tf.name)
    cap = cv2.VideoCapture(str(tmp))
    ok, frame = cap.read()
    cap.release()
    tmp.unlink(missing_ok=True)
    return frame if ok else None


@st.cache_data(show_spinner=False)
def read_output_bytes(path: str) -> bytes:
    """Cached read so the download button doesn't reload the whole video on
    every rerun (e.g. each slider nudge)."""
    return Path(path).read_bytes()


def iou_box(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def find_target_by_box(result, drag_box, allow_centre_fallback=True):
    """Return the track ID whose box best overlaps the drag box.

    Returns ``None`` when no tracked box overlaps the drag box, unless
    ``allow_centre_fallback`` is set, in which case the nearest-centre track is
    returned as a last resort. Callers acquire over the first few frames with
    the fallback disabled so a not-yet-detected target isn't mistaken for the
    nearest unrelated object on frame 0.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best_id, best_iou = None, 0.0
    for b in result.boxes:
        if b.id is None:
            continue
        box = b.xyxy[0].cpu().numpy().astype(float)
        score = iou_box(box, drag_box)
        if score > best_iou:
            best_iou, best_id = score, int(b.id)
    if best_id is not None or not allow_centre_fallback:
        return best_id
    # Fallback: nearest centre to drag-box centre
    dc = np.array([(drag_box[0]+drag_box[2])/2, (drag_box[1]+drag_box[3])/2])
    min_dist, best_id = float("inf"), None
    for b in result.boxes:
        if b.id is None:
            continue
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(float)
        c = np.array([(x1+x2)/2, (y1+y2)/2])
        d = float(np.linalg.norm(dc - c))
        if d < min_dist:
            min_dist, best_id = d, int(b.id)
    return best_id


def plot_single(result, target_id):
    frame = result.orig_img.copy()
    if result.boxes is None:
        return frame
    for b in result.boxes:
        if b.id is None or int(b.id) != target_id:
            continue
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
        label = f"id:{target_id} {result.names[int(b.cls)]} {float(b.conf):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def make_preview_figure(rgb, drag_box=None):
    """Plotly figure of the frame with optional selection rectangle overlay."""
    fig = px.imshow(rgb)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        dragmode="select",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="y", constrain="domain"),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   constrain="domain"),
        newselection=dict(line=dict(color="#3fb950", width=2)),
        activeselection=dict(fillcolor="rgba(63,185,80,0.15)"),
    )
    # Draw the confirmed box as a persistent green shape
    if drag_box:
        x0, y0, x1, y1 = drag_box
        fig.add_shape(type="rect",
                      x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="#3fb950", width=2),
                      fillcolor="rgba(63,185,80,0.12)")
    fig.update_traces(hoverinfo="skip", hovertemplate=None)
    return fig


def _reset_target_selection():
    """Drop the drag box and force the preview chart to a fresh widget (new
    key) so its persisted selection can't immediately re-populate the box on
    the same rerun (which silently defeated the 'Clear target' button)."""
    st.session_state.pop("drag_box", None)
    st.session_state["chart_gen"] = st.session_state.get("chart_gen", 0) + 1


def _on_file_change():
    """New (or cleared) upload: forget any target and prior output so a stale
    box / annotated video from a previous video isn't reused."""
    _reset_target_selection()
    st.session_state.pop("last_out_path", None)


# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-title">🎯 Video Object Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">YOLO11 · Real-time tracking</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload", type=["mp4", "mov", "avi", "mkv", "webm"],
                                label_visibility="collapsed",
                                on_change=_on_file_change)

    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"],
                              label_visibility="collapsed",
                              help="Nano = fastest · Large = most accurate")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-label">Tracker</div>', unsafe_allow_html=True)
        tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"],
                               label_visibility="collapsed")
    with c2:
        st.markdown('<div class="section-label">Confidence</div>', unsafe_allow_html=True)
        conf = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">Distortion</div>', unsafe_allow_html=True)
    distortion_on = st.toggle("Enable distortion", value=False)
    if distortion_on:
        bright = st.slider("Brightness gain", 0.2, 3.0, 1.0, 0.05,
                           help="<1 darkens · >1 brightens · 1.0 = off")
        mblur  = st.slider("Motion blur (px)", 0, 31, 0, 1,
                           help="Horizontal streak length · 0 = off")
        gblur  = st.slider("Gaussian blur (σ)", 0.0, 10.0, 0.0, 0.5,
                           help="Defocus radius · 0 = off")
    else:
        bright, mblur, gblur = 1.0, 0, 0.0

    st.markdown("---")

    # Target status + run button live in sidebar so they're always visible
    drag_box = st.session_state.get("drag_box")
    if drag_box:
        x0, y0, x1, y1 = [int(v) for v in drag_box]
        st.markdown(
            f'<div class="status-locked">🟢 Target selected ({x0},{y0}) → ({x1},{y1})</div>',
            unsafe_allow_html=True,
        )
        st.button("Clear target", use_container_width=True,
                  on_click=_reset_target_selection)
    else:
        st.markdown('<div class="status-none">⚪ No target — tracks all objects</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("▶ Run tracking", type="primary", use_container_width=True,
                    disabled=uploaded is None)


# ── main panel ─────────────────────────────────────────────────────────────────

if not uploaded:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:70vh;color:#8b949e;text-align:center;">
        <div style="font-size:3rem;margin-bottom:1rem;">🎬</div>
        <div style="font-size:1.1rem;font-weight:600;color:#c9d1d9;margin-bottom:0.5rem;">
            Upload a video to get started
        </div>
        <div style="font-size:0.82rem;">
            Use the sidebar to upload your file, choose a model, and configure settings.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

data = uploaded.getvalue()
raw_frame = get_first_frame(data)

if raw_frame is None:
    st.error("Could not decode the first frame of this video.")
    st.stop()

# Apply distortion to preview
distorted = apply_distortion(raw_frame, bright, mblur, gblur)
rgb = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)

drag_box = st.session_state.get("drag_box")

left_col, right_col = st.columns(2)

with left_col:
    st.markdown('<div class="panel-heading">Drag to select target · leave empty for all objects</div>',
                unsafe_allow_html=True)
    fig = make_preview_figure(rgb, drag_box)
    # Key includes a generation counter so 'Clear target' / a new upload reset
    # the widget's persisted selection instead of re-applying a stale box.
    chart_key = f"preview_chart_{st.session_state.get('chart_gen', 0)}"
    sel = st.plotly_chart(fig, on_select="rerun", selection_mode=["box"],
                          use_container_width=True, key=chart_key,
                          config={"displayModeBar": False, "scrollZoom": False})
    # Parse drag-box from Plotly selection event
    try:
        box = sel.selection["box"][0]
        xs = sorted(box["x"]); ys = sorted(box["y"])
        st.session_state["drag_box"] = [xs[0], ys[0], xs[1], ys[1]]
    except (KeyError, IndexError, TypeError):
        pass

# The right column is filled through a placeholder so a freshly-produced video
# can be shown in place right after tracking, without a full st.rerun() (which
# flashes and discards the success banner).
output_slot = right_col.empty()


def render_output():
    out_path = st.session_state.get("last_out_path")
    if out_path and Path(out_path).exists():
        with output_slot.container():
            st.markdown('<div class="panel-heading">Annotated output</div>',
                        unsafe_allow_html=True)
            st.video(out_path)
            st.download_button(
                "⬇ Download annotated video",
                read_output_bytes(out_path),
                file_name="annotated.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
    else:
        with output_slot.container():
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:100%;min-height:300px;border:1px dashed #30363d;
                        border-radius:8px;color:#8b949e;font-size:0.82rem;">
                Annotated video will appear here after running
            </div>
            """, unsafe_allow_html=True)


render_output()

# ── run tracking ───────────────────────────────────────────────────────────────

if run:
    # One temp dir per run; the previous run's dir is removed at the end (after
    # its output is no longer on screen) so vidtrack_* dirs don't pile up in
    # $TMPDIR across runs.
    prev_workdir = st.session_state.get("workdir")
    workdir = Path(tempfile.mkdtemp(prefix="vidtrack_"))
    st.session_state["workdir"] = str(workdir)
    in_path = workdir / uploaded.name
    in_path.write_bytes(data)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        cap.release()
        st.error("Could not open video — corrupt file or unsupported codec.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        st.error("Could not read frame dimensions from this video.")
        st.stop()

    out_path = workdir / "annotated.mp4"
    codec_used = None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
        )
        if writer.isOpened():
            codec_used = fourcc
            break
        writer.release()
    if codec_used is None:
        cap.release()
        st.error("Could not initialise the video writer.")
        st.stop()

    with st.spinner(f"Loading {model_name}…"):
        model = YOLO(model_name)

    target_id = None
    progress = st.progress(0.0, text="Tracking…")
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = apply_distortion(frame, bright, mblur, gblur)
            results = model.track(frame, persist=True, tracker=tracker,
                                  conf=conf, verbose=False)
            if drag_box:
                # Acquire the target over the first few frames: ByteTrack may
                # not assign the boxed object an ID on frame 0, so retry by real
                # overlap, only allowing the nearest-centre fallback once the
                # window is exhausted.
                if target_id is None and frame_idx < ACQUIRE_WINDOW:
                    target_id = find_target_by_box(
                        results[0], drag_box,
                        allow_centre_fallback=(frame_idx == ACQUIRE_WINDOW - 1),
                    )
                annotated = (plot_single(results[0], target_id)
                             if target_id is not None else results[0].plot())
            else:
                annotated = results[0].plot()

            writer.write(annotated)
            frame_idx += 1
            if total_frames > 0:
                progress.progress(min(frame_idx / total_frames, 1.0),
                                  text=f"Frame {frame_idx} / {total_frames}")
    finally:
        cap.release()
        writer.release()
        progress.empty()

    if frame_idx == 0:
        st.error("No frames could be decoded — nothing to annotate.")
        st.stop()

    # Ensure the result is H.264 so st.video plays it in the browser. If the
    # avc1 encoder was unavailable we wrote mp4v (not HTML5-playable); re-encode
    # with ffmpeg when present, otherwise warn instead of failing silently.
    if codec_used != "avc1":
        if shutil.which("ffmpeg"):
            h264_path = workdir / "annotated_h264.mp4"
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", str(out_path),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(h264_path)],
                capture_output=True,
            )
            if proc.returncode == 0 and h264_path.exists():
                out_path = h264_path
            else:
                st.warning("ffmpeg re-encode failed; the video may not play "
                           "in-browser — use the download button.")
        else:
            st.warning("Wrote video with the mp4v codec (no H.264 encoder or "
                       "ffmpeg available); it may not play in-browser — use the "
                       "download button below.")

    if drag_box and target_id is None:
        st.warning("Couldn’t lock onto the boxed object — showing all objects.")

    label = f"id:{target_id}" if target_id is not None else "all objects"
    st.session_state["last_out_path"] = str(out_path)
    render_output()
    st.success(f"Done — {frame_idx} frames · tracking {label}")

    # Now that the new output is rendered, drop the previous run's temp dir.
    if prev_workdir and prev_workdir != str(workdir) and Path(prev_workdir).exists():
        shutil.rmtree(prev_workdir, ignore_errors=True)
