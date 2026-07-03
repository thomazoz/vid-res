"""How well does detection survive blur and lighting changes on *video*?

Still images can't show the "objects in motion" angle, so this runs on real
video. The clips are unlabeled, so we use the detector's output on the *clean*
video as pseudo-ground-truth ("what we reliably see when conditions are good"),
then apply increasing motion blur and lighting changes and measure how much of
that survives:

  retention   - fraction of clean-frame objects still detected (recall proxy)
  mean conf   - average confidence of the surviving detections
  mean IoU    - how well the surviving boxes still localize (vs the clean box)
  det / frame - raw detections per frame

Outputs: CSV, a 2-panel plot (vs blur, vs lighting), montage images showing a
busy frame under each condition, and annotated worst-case videos for QuickTime.

    python3 video_robustness.py samples/people.mp4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corruptions import motion_blur, brightness

MODEL = "yolo11n.pt"
CONF = 0.3
IOU_THR = 0.5
STRIDE = 2  # sample every Nth frame for the metric sweep (speed)

BLUR_LEVELS = [0, 9, 17, 25]        # motion-blur length in px  (0 = clean)
LIGHT_LEVELS = [0.3, 0.6, 1.0, 1.8]  # brightness gain          (1.0 = clean)


# ----------------------------------------------------------- detection helpers
def detect(model, frame):
    """Return list of (xyxy ndarray, cls int, conf float) for one frame."""
    r = model.predict(frame, conf=CONF, verbose=False)[0]
    out = []
    if r.boxes is not None:
        for b in r.boxes:
            out.append((b.xyxy[0].cpu().numpy(), int(b.cls), float(b.conf)))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(ref, det):
    """Greedy same-class IoU matching of detections `det` to reference `ref`.
    Returns (n_matched, list_of_match_ious, list_of_match_confs)."""
    used, n, ious, confs = set(), 0, [], []
    for box, cls, conf in sorted(det, key=lambda d: -d[2]):  # high conf first
        best_j, best_i = -1, IOU_THR
        for j, (rbox, rcls, _) in enumerate(ref):
            if j in used or rcls != cls:
                continue
            i = iou(box, rbox)
            if i >= best_i:
                best_j, best_i = j, i
        if best_j >= 0:
            used.add(best_j)
            n += 1
            ious.append(best_i)
            confs.append(conf)
    return n, ious, confs


# ----------------------------------------------------------- the sweep
def load_sampled_frames(path):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % STRIDE == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames


def sweep(model, frames, corruption_fn, levels, clean_level):
    """Build clean reference once, then evaluate each level against it."""
    # reference = clean detections per frame (pseudo-ground-truth)
    ref = [detect(model, f) for f in frames]
    ref_total = sum(len(r) for r in ref)
    rows = []
    for lvl in levels:
        tot_match, tot_ref, tot_det, all_iou, all_conf = 0, 0, 0, [], []
        for f, rb in zip(frames, ref):
            if not rb:
                continue  # only score frames that had objects when clean
            cf = f if lvl == clean_level else corruption_fn(f, lvl)
            det = detect(model, cf)
            m, ious, confs = match(rb, det)
            tot_match += m
            tot_ref += len(rb)
            tot_det += len(det)
            all_iou += ious
            all_conf += confs
        rows.append({
            "level": lvl,
            "retention": tot_match / tot_ref if tot_ref else 0.0,
            "mean_conf": float(np.mean(all_conf)) if all_conf else 0.0,
            "mean_iou": float(np.mean(all_iou)) if all_iou else 0.0,
            "det_per_frame": tot_det / len(frames),
        })
        print(f"    level={lvl:<5} retention={rows[-1]['retention']:.3f} "
              f"conf={rows[-1]['mean_conf']:.3f} iou={rows[-1]['mean_iou']:.3f}")
    return rows, ref, ref_total


# ----------------------------------------------------------- visuals
def montage(model, frame, corruption_fn, levels, label_fn, out_path):
    tiles = []
    for lvl in levels:
        cf = corruption_fn(frame, lvl)
        r = model.predict(cf, conf=CONF, verbose=False)[0]
        plotted = r.plot()
        n = 0 if r.boxes is None else len(r.boxes)
        cv2.putText(plotted, f"{label_fn(lvl)} | {n} det", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tiles.append(plotted)
    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    cv2.imwrite(str(out_path), cv2.hconcat(tiles))
    print(f"  saved montage -> {out_path}")


def annotated_video(model, in_path, corruption_fn, level, out_path):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cf = corruption_fn(f, level)
        wr.write(model.predict(cf, conf=CONF, verbose=False)[0].plot())
    cap.release()
    wr.release()
    print(f"  saved video -> {out_path}")


def plot(blur_rows, light_rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, rows, xlabel, clean in [
        (axes[0], blur_rows, "Motion blur length (px)", 0),
        (axes[1], light_rows, "Brightness gain (x)", 1.0),
    ]:
        xs = [r["level"] for r in rows]
        ax.plot(xs, [r["retention"] for r in rows], "o-", label="object retention")
        ax.plot(xs, [r["mean_conf"] for r in rows], "s-", label="mean confidence")
        ax.plot(xs, [r["mean_iou"] for r in rows], "^-", label="mean IoU (localization)")
        ax.axvline(clean, color="gray", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("score (vs clean video)")
    axes[0].set_title("Effect of blur on moving-object detection")
    axes[1].set_title("Effect of lighting on moving-object detection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="samples/people.mp4")
    ap.add_argument("--outdir", default="video_robustness_out")
    ap.add_argument("--make-videos", action="store_true",
                    help="also render worst-case annotated videos (slower)")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    stem = Path(args.video).stem

    model = YOLO(MODEL)
    print(f"Loading + sampling frames from {args.video} (every {STRIDE})...")
    frames = load_sampled_frames(args.video)
    print(f"  {len(frames)} frames sampled")

    print("\n[BLUR sweep]")
    blur_rows, ref, ref_total = sweep(model, frames, motion_blur, BLUR_LEVELS, 0)
    print("[LIGHTING sweep]")
    light_rows, _, _ = sweep(model, frames, brightness, LIGHT_LEVELS, 1.0)
    print(f"\n(pseudo-ground-truth: {ref_total} objects across "
          f"{sum(1 for r in ref if r)} object-containing frames)")

    # write csv
    with open(out / f"{stem}_video_robustness.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "level", "retention", "mean_conf", "mean_iou", "det_per_frame"])
        for r in blur_rows:
            w.writerow(["blur", r["level"], r["retention"], r["mean_conf"], r["mean_iou"], r["det_per_frame"]])
        for r in light_rows:
            w.writerow(["light", r["level"], r["retention"], r["mean_conf"], r["mean_iou"], r["det_per_frame"]])

    plot(blur_rows, light_rows, out / f"{stem}_video_robustness.png")

    # montage on the busiest frame
    busy = max(range(len(frames)), key=lambda i: len(ref[i]))
    print(f"\nMontages on busiest frame (#{busy}, {len(ref[busy])} clean objects):")
    montage(model, frames[busy], motion_blur, BLUR_LEVELS,
            lambda l: f"blur={l}px", out / f"{stem}_montage_blur.png")
    montage(model, frames[busy], brightness, LIGHT_LEVELS,
            lambda l: f"gain={l}x", out / f"{stem}_montage_light.png")

    if args.make_videos:
        print("\nRendering worst-case annotated videos...")
        annotated_video(model, args.video, motion_blur, max(BLUR_LEVELS),
                        out / f"{stem}_blur{max(BLUR_LEVELS)}.mp4")
        annotated_video(model, args.video, brightness, min(LIGHT_LEVELS),
                        out / f"{stem}_dark{min(LIGHT_LEVELS)}.mp4")


if __name__ == "__main__":
    main()
