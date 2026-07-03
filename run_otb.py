"""Run the YOLO tracker on an OTB image sequence and save an annotated MP4.

By default this now tracks ONE thing — the sequence's ground-truth target.
The first frame's GT box picks which track to follow; if the tracker loses
that ID (common — see docs/HOW_THE_TRACKER_WORKS.md), the follower re-locks
onto the best-overlapping track and counts the switch. Use --all-objects to
draw every tracked object like before.

Usage:
    python3 run_otb.py Basketball
    python3 run_otb.py Car1 --model yolo11s.pt --conf 0.3
    python3 run_otb.py Crowds --show-gt          # overlay ground-truth box in blue
    python3 run_otb.py Basketball --all-objects  # old behavior: annotate everything
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from single_target import TargetFollower

OTB_DIR = Path("ds/OTB-dataset/OTB_downloads")
OUT_DIR = Path("otb_runs")

TARGET_GREEN = (60, 220, 60)
LOST_RED = (60, 60, 230)
GT_BLUE = (255, 100, 0)


def load_gt(seq_dir):
    gt_path = seq_dir / "groundtruth_rect.txt"
    if not gt_path.exists():
        return []
    gt = []
    with open(gt_path) as f:
        for line in f:
            parts = [p for p in line.strip().replace("\t", ",").replace(" ", ",").split(",") if p]
            if len(parts) >= 4:
                x, y, w, h = (int(float(p)) for p in parts[:4])
                gt.append((x, y, x + w, y + h))
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence", help="OTB sequence name, e.g. Basketball")
    ap.add_argument("--model",   default="yolo11n.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf",    type=float, default=0.25)
    ap.add_argument("--show-gt", action="store_true", help="Overlay GT box in blue")
    ap.add_argument("--all-objects", action="store_true",
                    help="Annotate every tracked object (old behavior) instead of "
                         "following only the GT target")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--outdir",  default=str(OUT_DIR))
    args = ap.parse_args()

    seq_dir = OTB_DIR / args.sequence
    if not seq_dir.exists():
        print(f"Sequence not found: {seq_dir}")
        print("Available sequences:")
        for p in sorted(OTB_DIR.iterdir()):
            if p.is_dir():
                print(f"  {p.name}")
        return

    img_files = sorted((seq_dir / "img").iterdir())
    if args.max_frames > 0:
        img_files = img_files[:args.max_frames]
    gt = load_gt(seq_dir)
    if not args.all_objects and not gt:
        print("No ground truth found — falling back to --all-objects mode.")
        args.all_objects = True

    if not img_files:
        print("No images found.")
        return

    f0 = cv2.imread(str(img_files[0]))
    h, w = f0.shape[:2]
    fps = 30

    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.sequence}_tracked.mp4"

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(args.model)
    follower = TargetFollower()
    trail: list[tuple[int, int]] = []
    mode = "all objects" if args.all_objects else "single target (GT-seeded)"
    print(f"\nTracking {args.sequence}  ({len(img_files)} frames, {mode})  → {out_path}\n")

    for i, img_path in enumerate(img_files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = model.track(frame, persist=True, tracker=args.tracker,
                              conf=args.conf, verbose=False)
        boxes = results[0].boxes
        gt_box = gt[i] if i < len(gt) else None

        if args.all_objects:
            annotated = results[0].plot()
        else:
            annotated = frame.copy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes is not None and boxes.id is not None else []
            xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and len(boxes) else np.zeros((0, 4))
            clss = boxes.cls.cpu().numpy().astype(int) if boxes is not None and len(boxes) else None
            confs = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) else []

            tbox, tid, status = follower.update(ids, xyxy, clss, ref_box=gt_box)
            if tbox is not None:
                x1, y1, x2, y2 = (int(v) for v in tbox)
                trail.append(((x1 + x2) // 2, (y1 + y2) // 2))
                k = list(ids).index(tid)
                cls_name = model.names[int(clss[k])] if clss is not None else "?"
                label = f"TARGET id:{tid} {cls_name} {confs[k]:.2f}"
                if status == "reacquired":
                    label += "  (re-locked)"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), TARGET_GREEN, 3)
                cv2.putText(annotated, label, (x1, max(16, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TARGET_GREEN, 2, cv2.LINE_AA)
            elif status == "lost":
                cv2.putText(annotated, "target lost...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, LOST_RED, 2, cv2.LINE_AA)
            for p, q in zip(trail[-60:], trail[-59:]):
                cv2.line(annotated, p, q, TARGET_GREEN, 2)

        if args.show_gt and gt_box is not None:
            x1, y1, x2, y2 = gt_box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), GT_BLUE, 2)
            cv2.putText(annotated, "GT", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, GT_BLUE, 2)

        writer.write(annotated)

        if (i + 1) % 50 == 0 or i == len(img_files) - 1:
            print(f"  {i+1}/{len(img_files)} frames")

    writer.release()
    if not args.all_objects:
        print(f"\n{follower.summary()}")
    print(f"\nSaved → {out_path.resolve()}")
    print("Open in QuickTime or view it from the hub's OTB page.")


if __name__ == "__main__":
    main()
