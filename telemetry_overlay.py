"""Burn live tracker telemetry into a video and write an xlsx report.

Runs YOLO11 + ByteTrack over a video file or an OTB image sequence and, for
every frame, renders a semi-transparent HUD panel with live statistics
(detections, active track IDs with births/deaths, confidence, brightness,
and — in OTB mode — per-frame IoU vs ground truth plus running accuracy).
The annotated boxes+IDs stay on the frame; in OTB mode the ground-truth box
is drawn in blue.

OTB accuracy follows otb_eval.py: on the first frame where a track overlaps
the ground truth, lock onto that track ID and follow it; per-frame
IoU = IoU(GT, followed box) (0 when the ID is absent), hit = IoU >= 0.5,
running accuracy = fraction of frames so far with a hit.

Outputs (in --outdir, default telemetry_out/):
    <stem>_telemetry.mp4   video with the HUD burned in
    <stem>_report.xlsx     sheets: per_frame, summary, tracks
    <stem>_timeline.png    detections (and IoU/accuracy in OTB mode) per frame

Usage:
    python3 telemetry_overlay.py samples/people.mp4
    python3 telemetry_overlay.py samples/traffic.mp4 --max-frames 300 --conf 0.4
    python3 telemetry_overlay.py --seq Basketball --max-frames 200
    python3 telemetry_overlay.py --seq Bolt --model yolo11l.pt --tracker botsort.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from single_target import TargetFollower

OTB_DIR = Path("ds/OTB-dataset/OTB_downloads")
OTB_FPS = 30.0          # OTB sequences are plain image folders; assume 30 fps
CLIP_LUMA = 250         # Y >= this counts as clipped-highlight
DARK_LUMA, BRIGHT_LUMA = 70.0, 180.0


def auto_device(pref: str | None = None) -> str:
    """Resolve the inference device: honour an explicit choice, else prefer
    Apple MPS when available, else fall back to CPU."""
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ── small helpers ─────────────────────────────────────────────────────────────

def load_gt(seq_dir: Path) -> list[tuple[int, int, int, int]]:
    """Ground-truth (x, y, w, h) boxes, one per frame; separators vary."""
    gt = []
    with open(seq_dir / "groundtruth_rect.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p]
            x, y, w, h = (int(float(v)) for v in parts[:4])
            gt.append((x, y, w, h))
    return gt


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def frame_luma(frame: np.ndarray) -> tuple[float, float]:
    """(mean BT.709 luma, % of pixels with Y >= CLIP_LUMA). Subsampled for speed."""
    small = frame[::4, ::4].astype(np.float32)         # BGR
    y = 0.0722 * small[..., 0] + 0.7152 * small[..., 1] + 0.2126 * small[..., 2]
    return float(y.mean()), float((y >= CLIP_LUMA).mean() * 100.0)


def luma_tag(y: float) -> str:
    if y < DARK_LUMA:
        return "dark"
    if y > BRIGHT_LUMA:
        return "bright"
    return "ok"


# ── HUD rendering ─────────────────────────────────────────────────────────────

WHITE  = (235, 235, 235)
AMBER  = (60, 200, 255)
GREEN  = (90, 220, 90)
RED    = (70, 70, 235)
GT_BLUE = (255, 130, 0)


def draw_hud(frame: np.ndarray, rows: list[tuple[str, tuple]]) -> np.ndarray:
    """Semi-transparent dark panel in the top-right corner with text rows."""
    h, w = frame.shape[:2]
    s = min(max(w / 1280.0, 0.55), 1.2)                # size factor for 640–1280px
    fs = max(0.42, 0.52 * s)
    th = 1 if fs < 0.7 else 2
    line_h = int(30 * s) + 6
    pad = int(12 * s) + 2
    panel_w = max(int(340 * s), 200)
    panel_h = pad * 2 + line_h * len(rows)
    x0, y0 = w - panel_w - int(8 * s), int(8 * s)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (25, 22, 18), -1)
    frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (90, 90, 90), 1)

    for i, (text, colour) in enumerate(rows):
        yy = y0 + pad + line_h * i + int(line_h * 0.72)
        cv2.putText(frame, text, (x0 + pad, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, colour, th, cv2.LINE_AA)
    return frame


# ── frame sources ─────────────────────────────────────────────────────────────

def video_frames(path: Path, max_frames: int | None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps = fps if fps > 1 else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames:
        total = min(total, max_frames) if total else max_frames

    def gen():
        i = 0
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, None
            i += 1
        cap.release()
    return gen(), fps, total


def otb_frames(seq_dir: Path, max_frames: int | None):
    gt_list = load_gt(seq_dir)
    imgs = sorted((seq_dir / "img").glob("*.jpg"))
    n = min(len(gt_list), len(imgs))
    if max_frames:
        n = min(n, max_frames)

    def gen():
        for i in range(n):
            frame = cv2.imread(str(imgs[i]))
            if frame is None:
                continue
            yield frame, gt_list[i]
    return gen(), OTB_FPS, n


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(args) -> None:
    device = auto_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    if args.seq:
        seq_dir = OTB_DIR / args.seq
        if not (seq_dir / "groundtruth_rect.txt").exists():
            raise SystemExit(f"No ground truth at {seq_dir}")
        frames, fps, total = otb_frames(seq_dir, args.max_frames)
        stem, source_name, otb = args.seq, str(seq_dir), True
    else:
        src = Path(args.source)
        frames, fps, total = video_frames(src, args.max_frames)
        stem, source_name, otb = src.stem, str(src), False

    print(f"Source: {source_name}  |  Model: {args.model}  |  Tracker: {args.tracker}")
    print(f"Conf: {args.conf}  |  Device: {device}  |  Frames planned: {total or '?'}\n")

    model = YOLO(args.model)
    names = model.names

    writer = None
    video_path = outdir / f"{stem}_telemetry.mp4"
    per_frame: list[dict] = []
    tracks: dict[int, dict] = {}           # id -> first/last/n/conf_sum/cls counts
    prev_ids: set[int] = set()
    followed_id = None
    follower = TargetFollower() if args.single else None
    trail: list[tuple[int, int]] = []
    hits = 0
    frame_idx = 0
    fw = fh = 0

    for frame, gt in frames:
        if writer is None:
            fh, fw = frame.shape[:2]
            writer = cv2.VideoWriter(str(video_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

        t0 = time.perf_counter()
        results = model.track(frame, persist=True, conf=args.conf,
                              tracker=args.tracker, verbose=False, device=device)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        r = boxes = results[0].boxes
        n_det = len(boxes) if boxes is not None else 0
        confs = boxes.conf.cpu().numpy() if n_det else np.array([])
        xyxy = boxes.xyxy.cpu().numpy() if n_det else np.zeros((0, 4))
        clss = boxes.cls.cpu().numpy().astype(int) if n_det else np.array([], int)
        ids = (boxes.id.cpu().numpy().astype(int)
               if n_det and boxes.id is not None else np.array([], int))

        id_set = set(int(i) for i in ids)
        new_ids, lost_ids = id_set - prev_ids, prev_ids - id_set
        prev_ids = id_set

        for k, tid in enumerate(ids):
            tid = int(tid)
            t = tracks.setdefault(tid, {"first": frame_idx, "last": frame_idx,
                                        "n": 0, "conf_sum": 0.0, "cls": {}})
            t["last"] = frame_idx
            t["n"] += 1
            t["conf_sum"] += float(confs[k]) if k < len(confs) else 0.0
            cname = names.get(int(clss[k]), str(clss[k])) if k < len(clss) else "?"
            t["cls"][cname] = t["cls"].get(cname, 0) + 1

        areas_pct = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                     / (fw * fh) * 100.0) if n_det else np.array([])
        luma, clip_pct = frame_luma(frame)

        row = {
            "frame_idx": frame_idx,
            "time_s": round(frame_idx / fps, 3),
            "n_detections": n_det,
            "n_active_ids": len(id_set),
            "active_track_ids": ",".join(str(i) for i in sorted(id_set)),
            "n_new_ids": len(new_ids),
            "n_lost_ids": len(lost_ids),
            "mean_conf": round(float(confs.mean()), 4) if n_det else np.nan,
            "min_conf": round(float(confs.min()), 4) if n_det else np.nan,
            "mean_box_area_pct": round(float(areas_pct.mean()), 3) if n_det else np.nan,
            "brightness_mean_luma": round(luma, 2),
            "clip_high_pct": round(clip_pct, 3),
            "inference_ms": round(infer_ms, 2),
        }

        gt_xy = None
        if otb:
            gx, gy, gw, gh = gt
            gt_xy = (gx, gy, gx + gw, gy + gh)

        target_box = target_status = None
        if follower is not None:
            target_box, followed_id, target_status = follower.update(
                ids, xyxy, clss, ref_box=gt_xy)

        iou = None
        if otb:
            if follower is not None:                    # single mode: follow with re-lock
                pred = target_box
            else:                                       # strict lock (otb_eval semantics)
                if followed_id is None:                 # acquire on first overlap
                    best = 0.0
                    for k, tid in enumerate(ids):
                        i = iou_xyxy(xyxy[k], gt_xy)
                        if i > best:
                            best, followed_id = i, int(tid)
                pred = None
                for k, tid in enumerate(ids):
                    if followed_id is not None and int(tid) == followed_id:
                        pred = xyxy[k]
                        break
            iou = iou_xyxy(pred, gt_xy) if pred is not None else 0.0
            hit = int(iou >= 0.5)
            hits += hit
            running_acc = hits / (frame_idx + 1)
            row.update({"gt_x": gx, "gt_y": gy, "gt_w": gw, "gt_h": gh,
                        "followed_id": followed_id if followed_id is not None else np.nan,
                        "iou": round(iou, 4), "hit": hit,
                        "running_accuracy": round(running_acc, 4)})
        per_frame.append(row)

        # ── render ──
        if follower is None:
            annotated = results[0].plot()
        else:                                          # single-target rendering
            annotated = frame.copy()
            if target_box is not None:
                x1, y1, x2, y2 = (int(v) for v in target_box)
                trail.append(((x1 + x2) // 2, (y1 + y2) // 2))
                k = list(ids).index(followed_id)
                cls_name = names.get(int(clss[k]), "?") if len(clss) else "?"
                label = f"TARGET id:{followed_id} {cls_name}"
                if target_status == "reacquired":
                    label += " (re-locked)"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), GREEN, 3)
                cv2.putText(annotated, label, (x1, max(16, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
            elif target_status == "lost":
                cv2.putText(annotated, "target lost...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
            for p, q in zip(trail[-60:], trail[-59:]):
                cv2.line(annotated, p, q, GREEN, 2)
        if otb:
            cv2.rectangle(annotated, (gt_xy[0], gt_xy[1]), (gt_xy[2], gt_xy[3]),
                          GT_BLUE, 2)
            cv2.putText(annotated, "GT", (gt_xy[0], max(14, gt_xy[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_BLUE, 1, cv2.LINE_AA)

        hud = [
            ("TELEMETRY", AMBER),
            (f"frame {frame_idx}  t={frame_idx / fps:.1f}s", WHITE),
            (f"det {n_det}", WHITE),
            (f"ids {len(id_set)}  +{len(new_ids)}/-{len(lost_ids)}", WHITE),
        ]
        if follower is not None:
            hud.append((f"target id:{followed_id if followed_id is not None else '-'}"
                        f"  relocks {follower.n_switches}",
                        GREEN if target_box is not None else RED))
        hud += [
            (f"conf {row['mean_conf']:.2f}" if n_det else "conf -", WHITE),
            (f"luma {luma:.0f} {luma_tag(luma)}",
             WHITE if luma_tag(luma) == "ok" else AMBER),
        ]
        if otb:
            hud.append((f"IoU {iou:.2f}", GREEN if iou >= 0.5 else RED))
            hud.append((f"acc {running_acc * 100:.1f}%", WHITE))
        writer.write(draw_hud(annotated, hud))

        if frame_idx % 30 == 0:
            extra = f"  iou={iou:.2f}" if iou is not None else ""
            print(f"  frame {frame_idx:4d}/{total or '?'}  det={n_det:2d}  "
                  f"ids={len(id_set):2d}  {infer_ms:6.1f}ms{extra}")
        frame_idx += 1

    if writer is not None:
        writer.release()
    if frame_idx == 0:
        raise SystemExit("No frames processed.")

    # ── report ──
    df = pd.DataFrame(per_frame)

    track_rows = []
    for tid in sorted(tracks):
        t = tracks[tid]
        top_cls = max(t["cls"], key=t["cls"].get) if t["cls"] else "?"
        track_rows.append({
            "track_id": tid,
            "first_frame": t["first"],
            "last_frame": t["last"],
            "length": t["last"] - t["first"] + 1,
            "n_frames_present": t["n"],
            "mean_conf": round(t["conf_sum"] / t["n"], 4) if t["n"] else np.nan,
            "class": top_cls,
        })
    tracks_df = pd.DataFrame(track_rows)

    mean_track_len = float(tracks_df["n_frames_present"].mean()) if len(tracks_df) else 0.0
    summary_items = [
        ("model", args.model),
        ("tracker", args.tracker),
        ("conf_threshold", args.conf),
        ("source", source_name),
        ("frames_processed", frame_idx),
        ("source_fps", round(fps, 3)),
        ("frame_size", f"{fw}x{fh}"),
        ("mean_detections_per_frame", round(float(df["n_detections"].mean()), 3)),
        ("unique_track_ids", len(tracks_df)),
        ("mean_track_length_frames", round(mean_track_len, 2)),
        ("mean_conf", round(float(df["mean_conf"].mean()), 4)),
        ("mean_brightness_luma", round(float(df["brightness_mean_luma"].mean()), 2)),
        ("mean_inference_ms", round(float(df["inference_ms"].mean()), 2)),
    ]
    if otb:
        summary_items += [
            ("mean_iou", round(float(df["iou"].mean()), 4)),
            ("success_rate_at_0.5", round(float(df["hit"].mean()), 4)),
            ("final_accuracy", round(hits / frame_idx, 4)),
        ]
    if follower is not None:
        summary_items += [
            ("single_target_mode", "yes (re-lock on loss)"),
            ("target_id_switches", follower.n_switches),
            ("target_frames_followed", follower.frames_tracked),
            ("target_frames_lost", follower.frames_lost),
        ]
    summary_df = pd.DataFrame(summary_items, columns=["metric", "value"])

    xlsx_path = outdir / f"{stem}_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="per_frame", index=False)
        summary_df.to_excel(xw, sheet_name="summary", index=False)
        tracks_df.to_excel(xw, sheet_name="tracks", index=False)

    # ── timeline plot ──
    png_path = outdir / f"{stem}_timeline.png"
    n_ax = 2 if otb else 1
    fig, axes = plt.subplots(n_ax, 1, figsize=(10, 3.2 * n_ax), sharex=True,
                             squeeze=False)
    ax = axes[0][0]
    ax.plot(df["frame_idx"], df["n_detections"], color="steelblue", lw=1.2)
    ax.set_ylabel("detections")
    ax.grid(alpha=0.3)
    ax.set_title(f"{stem} — telemetry timeline ({args.model}, {args.tracker})")
    if otb:
        ax2 = axes[1][0]
        ax2.plot(df["frame_idx"], df["iou"], color="seagreen", lw=1.0, label="IoU")
        ax2.plot(df["frame_idx"], df["running_accuracy"], color="darkorange",
                 lw=1.4, label="running acc")
        ax2.axhline(0.5, color="red", ls="--", lw=0.9)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("IoU / accuracy")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
    axes[-1][0].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    # ── closing summary ──
    print(f"\n{'=' * 62}")
    print(f"TELEMETRY SUMMARY — {stem}")
    print(f"  source        {source_name}  ({fw}x{fh} @ {fps:.2f} fps)")
    print(f"  frames        {frame_idx}")
    print(f"  det/frame     {df['n_detections'].mean():.2f}   "
          f"unique IDs {len(tracks_df)}   mean track len {mean_track_len:.1f}")
    print(f"  mean conf     {df['mean_conf'].mean():.3f}   "
          f"mean luma {df['brightness_mean_luma'].mean():.1f}   "
          f"infer {df['inference_ms'].mean():.1f} ms")
    if otb:
        print(f"  mean IoU      {df['iou'].mean():.3f}   "
              f"success@0.5 {df['hit'].mean():.3f}   "
              f"final acc {hits / frame_idx:.3f}   (followed ID {followed_id})")
    if follower is not None:
        print(f"  single-target {follower.summary()}")
    print(f"  video   -> {video_path}")
    print(f"  report  -> {xlsx_path}")
    print(f"  plot    -> {png_path}")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", nargs="?", default=None,
                    help="Path to a video file (omit when using --seq)")
    ap.add_argument("--seq", default=None,
                    help=f"OTB sequence name under {OTB_DIR} (uses ground truth)")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--single", action="store_true",
                    help="draw and score only ONE followed target (GT-seeded on OTB, "
                         "largest first detection otherwise); re-locks when the "
                         "tracker loses the ID and counts the switches")
    ap.add_argument("--outdir", default="telemetry_out")
    ap.add_argument("--device", default="auto",
                    help="Inference device: auto (MPS>CUDA>CPU), cpu, mps, cuda, 0…")
    args = ap.parse_args()

    if bool(args.source) == bool(args.seq):
        ap.error("Provide exactly one input: a video path OR --seq <OTBName>")
    run(args)


if __name__ == "__main__":
    main()
