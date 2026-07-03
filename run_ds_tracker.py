"""Run a tracker on every OTB sequence in ds/ and save outputs.

The Streamlit app (app.py) uses YOLO + ByteTrack for multi-object tracking.
We could not install ultralytics/torch in this sandbox (the wheels are ~500 MB
and don't fit in the available download window). Instead we run OpenCV's CSRT
single-object tracker initialized from each OTB sequence's first-frame
groundtruth_rect.txt box — which is the canonical way to evaluate trackers on
OTB. Outputs land in vid res/results/.

For each sequence we write:
  results/<seq>/annotated.mp4        annotated MP4 (H.264 if ffmpeg available)
  results/<seq>/predictions.csv      per-frame predicted box + GT + IoU
  results/<seq>/summary.json         frame count, mean IoU, success rate
And a top-level results/summary.csv aggregating all sequences.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

# Resolve paths relative to this script so it runs from any CWD.
BASE = Path(__file__).resolve().parent
SEQS_DIR = BASE / "ds" / "OTB-dataset" / "OTB_downloads"
OUT_DIR = BASE / "results"
TRACKER_NAME = "CSRT"


def make_tracker():
    return cv2.TrackerCSRT_create()


def load_gt(gt_path: Path) -> list[tuple[int, int, int, int]]:
    boxes = []
    for line in gt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # OTB uses either ',' or whitespace/tab as separator
        parts = [p for p in line.replace(",", " ").split() if p]
        x, y, w, h = (int(float(p)) for p in parts[:4])
        boxes.append((x, y, w, h))
    return boxes


def iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2); ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2); yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def reencode_h264(src: Path) -> Path:
    if not shutil.which("ffmpeg"):
        return src
    dst = src.with_name(src.stem + "_h264.mp4")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(dst)],
        capture_output=True,
    )
    if r.returncode == 0:
        try:
            src.unlink()
        except OSError:
            pass  # mount may not allow delete; harmless
        return dst
    return src


def run_sequence(seq_dir: Path, out_dir: Path) -> dict:
    name = seq_dir.name
    img_dir = seq_dir / "img"
    gt_path = seq_dir / "groundtruth_rect.txt"
    frames = sorted(img_dir.glob("*.jpg"))
    gt = load_gt(gt_path)
    if not frames or not gt:
        return {"sequence": name, "error": "no frames or no gt"}

    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = out_dir / "annotated.mp4"
    preds_path = out_dir / "predictions.csv"

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
    )

    tracker = make_tracker()
    init_box = tuple(gt[0])  # (x,y,w,h)
    tracker.init(first, init_box)

    rows = []
    ious = []
    success = 0
    t0 = time.time()
    n = min(len(frames), len(gt))
    for i in range(n):
        frame = first if i == 0 else cv2.imread(str(frames[i]))
        if frame is None:
            continue
        if i == 0:
            ok, box = True, init_box
        else:
            ok, box = tracker.update(frame)
            box = tuple(int(v) for v in box) if ok else (0, 0, 0, 0)

        gt_box = gt[i] if i < len(gt) else (0, 0, 0, 0)
        cur_iou = iou(box, gt_box) if ok else 0.0
        ious.append(cur_iou)
        if cur_iou >= 0.5:
            success += 1

        # Draw GT (green) and prediction (red)
        gx, gy, gw, gh = gt_box
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
        if ok:
            x, y, bw, bh = box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{name} f{i+1}/{n} IoU={cur_iou:.2f}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        writer.write(frame)
        rows.append({
            "frame": i + 1, "tracker_ok": int(ok),
            "pred_x": box[0], "pred_y": box[1], "pred_w": box[2], "pred_h": box[3],
            "gt_x": gx, "gt_y": gy, "gt_w": gw, "gt_h": gh,
            "iou": round(cur_iou, 4),
        })
    writer.release()
    dt = time.time() - t0

    with preds_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    final_video = reencode_h264(annotated_path)
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    summary = {
        "sequence": name,
        "tracker": TRACKER_NAME,
        "frames": n,
        "mean_iou": round(mean_iou, 4),
        "success_rate@0.5": round(success / n, 4),
        "throughput_fps": round(n / dt, 2) if dt > 0 else 0.0,
        "seconds": round(dt, 2),
        "annotated_video": str(final_video.relative_to(OUT_DIR.parent)),
        "predictions_csv": str(preds_path.relative_to(OUT_DIR.parent)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seqs = sorted(d for d in SEQS_DIR.iterdir()
                  if d.is_dir() and d.name != "__MACOSX"
                  and (d / "img").exists() and (d / "groundtruth_rect.txt").exists())
    print(f"Found {len(seqs)} sequences: {[s.name for s in seqs]}")
    only = set(sys.argv[1:])  # optional CLI filter
    summaries = []
    for seq in seqs:
        if only and seq.name not in only:
            continue
        print(f"\n=== {seq.name} ===")
        try:
            s = run_sequence(seq, OUT_DIR / seq.name)
            summaries.append(s)
            print(json.dumps(s, indent=2))
        except Exception as e:
            print(f"FAILED: {e}")
            summaries.append({"sequence": seq.name, "error": str(e)})

    # Top-level aggregate CSV
    agg_path = OUT_DIR / "summary.csv"
    keys = ["sequence", "tracker", "frames", "mean_iou", "success_rate@0.5",
            "throughput_fps", "seconds", "annotated_video", "predictions_csv", "error"]
    with agg_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader(); wr.writerows(summaries)
    print(f"\nWrote {agg_path}")


if __name__ == "__main__":
    main()
