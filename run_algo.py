"""Run the YOLO detection + tracking algorithm on a video (no Streamlit needed).

Mirrors app.py's pipeline as a CLI so you can run the algo headlessly:

    python3 run_algo.py samples/people.mp4
    python3 run_algo.py myvideo.mp4 --model yolo11s.pt --conf 0.3 --tracker bytetrack.yaml
    python3 run_algo.py samples/people.mp4 --single auto   # follow ONE object only
    python3 run_algo.py samples/people.mp4 --single 3      # follow track ID 3

Writes an annotated video next to the input and dumps a few sample frames so you
can eyeball the detections. With --single the output video follows a single
object (largest first detection, or a given track ID) and re-locks if the
tracker loses its ID.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from single_target import TargetFollower


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--outdir", default="run_out")
    ap.add_argument("--sample-frames", type=int, default=4, help="frames to dump as PNG")
    ap.add_argument("--single", default=None, metavar="auto|ID",
                    help="track only ONE object: 'auto' = largest first detection, "
                         "or a track ID number")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    args = ap.parse_args()

    follower = None
    if args.single is not None:
        want_id = None if args.single == "auto" else int(args.single)
        follower = TargetFollower(want_id=want_id)

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = outdir / f"{Path(args.video).stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit("Could not open VideoWriter")

    print(f"Input : {args.video}  ({width}x{height} @ {fps:.1f}fps, {total} frames)")
    print(f"Model : {args.model}   tracker={args.tracker}  conf={args.conf}")
    model = YOLO(args.model)

    # frames to snapshot, spread across the clip
    snap_at = set()
    if total > 0 and args.sample_frames > 0:
        step = max(total // (args.sample_frames + 1), 1)
        snap_at = {step * (i + 1) for i in range(args.sample_frames)}

    frame_idx, total_dets, track_ids = 0, 0, set()
    class_counts: dict[str, int] = {}
    trail: list[tuple[int, int]] = []
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break
        results = model.track(
            frame, persist=True, tracker=args.tracker, conf=args.conf, verbose=False
        )
        boxes = results[0].boxes
        if boxes is not None:
            total_dets += len(boxes)
            for c in boxes.cls:
                name = model.names[int(c)]
                class_counts[name] = class_counts.get(name, 0) + 1
            if boxes.id is not None:
                track_ids.update(int(i) for i in boxes.id)
        if follower is None:
            annotated = results[0].plot()
        else:
            annotated = frame.copy()
            has = boxes is not None and len(boxes) and boxes.id is not None
            ids = boxes.id.cpu().numpy().astype(int) if has else []
            xyxy = boxes.xyxy.cpu().numpy() if has else np.zeros((0, 4))
            clss = boxes.cls.cpu().numpy().astype(int) if has else None
            confs = boxes.conf.cpu().numpy() if has else []
            tbox, tid, status = follower.update(ids, xyxy, clss)
            if tbox is not None:
                x1, y1, x2, y2 = (int(v) for v in tbox)
                trail.append(((x1 + x2) // 2, (y1 + y2) // 2))
                k = list(ids).index(tid)
                cls_name = model.names[int(clss[k])] if clss is not None else "?"
                label = f"TARGET id:{tid} {cls_name} {confs[k]:.2f}"
                if status == "reacquired":
                    label += "  (re-locked)"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 60), 3)
                cv2.putText(annotated, label, (x1, max(16, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2, cv2.LINE_AA)
            elif status == "lost":
                cv2.putText(annotated, "target lost...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 230), 2, cv2.LINE_AA)
            for p, q in zip(trail[-60:], trail[-59:]):
                cv2.line(annotated, p, q, (60, 220, 60), 2)
        writer.write(annotated)
        if frame_idx in snap_at:
            cv2.imwrite(str(outdir / f"{Path(args.video).stem}_frame{frame_idx}.png"), annotated)
        frame_idx += 1
        if total > 0 and frame_idx % 25 == 0:
            print(f"  ...{frame_idx}/{total} frames", end="\r")
    dt = time.time() - t0

    cap.release()
    writer.release()

    top = sorted(class_counts.items(), key=lambda kv: -kv[1])[:8]
    print("\n========== RUN SUMMARY ==========")
    print(f"frames processed     : {frame_idx}")
    print(f"total detections     : {total_dets}  ({total_dets/max(frame_idx,1):.1f}/frame)")
    print(f"unique track IDs      : {len(track_ids)}")
    print(f"classes seen (top)    : {', '.join(f'{k}={v}' for k,v in top)}")
    print(f"throughput            : {frame_idx/dt:.1f} fps  ({dt:.1f}s total)")
    if follower is not None:
        print(f"single-target mode    : {follower.summary()}")
    print(f"annotated video       : {out_path}")


if __name__ == "__main__":
    main()
