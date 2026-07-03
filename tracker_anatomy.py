"""Make ByteTrack's internals visible: predictions, associations, births, deaths.

Runs YOLO detection and a real ultralytics BYTETracker side by side (the same
class model.track() uses — instantiated directly so its state is inspectable)
and renders, per frame:

    green thin boxes   raw YOLO detections (with confidence)
    yellow corners     each track's Kalman-PREDICTED box, before it sees
                       this frame's detections (the "where it should be" guess)
    solid colored box  the track after update, labeled id:N
    blue "NEW id"      a track born this frame
    red "LOST id"      a track that just went lost (kept for track_buffer frames)

Also writes a per-frame CSV of tracker state counts and a timeline plot —
watch detections drop and the lost-count spike when the target blurs or turns.

Usage:
    python3 tracker_anatomy.py                                   # samples/people.mp4
    python3 tracker_anatomy.py samples/traffic.mp4 --max-frames 200
    python3 tracker_anatomy.py --seq Basketball --max-frames 200 # OTB sequence
    python3 tracker_anatomy.py samples/people.mp4 --conf 0.1     # enable the low-conf 2nd pass
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

OTB_DIR = Path("ds/OTB-dataset/OTB_downloads")


def auto_device(pref: str = "auto") -> str:
    """auto → MPS > CUDA > CPU, so the tool also runs on machines without MPS."""
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


def make_tracker(tracker_yaml: str) -> BYTETracker:
    """Instantiate BYTETracker exactly like ultralytics/trackers/track.py does."""
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_yaml)))
    if cfg.tracker_type != "bytetrack":
        raise SystemExit("tracker_anatomy visualizes BYTETracker; use a bytetrack yaml")
    return BYTETracker(args=cfg, )


def frames_from(source: str | None, seq: str | None):
    if seq:
        seq_dir = OTB_DIR / seq / "img"
        files = sorted(seq_dir.glob("*.jpg"))
        if not files:
            raise SystemExit(f"no frames in {seq_dir}")
        for p in files:
            img = cv2.imread(str(p))
            if img is not None:
                yield img
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"could not open {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()


def color_for(tid: int):
    rng = np.random.default_rng(tid * 9973)
    return tuple(int(c) for c in rng.integers(80, 255, 3))


def draw_corners(img, box, color, length=10, thickness=2):
    """Corner-only rectangle (used for Kalman predictions)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", nargs="?", default="samples/people.mp4")
    ap.add_argument("--seq", default=None, help="OTB sequence name instead of a video file")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detector conf; set <= 0.1 to let ByteTrack's low-conf 2nd pass fire")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="tracker_anatomy_out")
    args = ap.parse_args()
    args.device = auto_device(args.device)

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    stem = args.seq if args.seq else Path(args.source).stem

    model = YOLO(args.model)
    tracker = make_tracker(args.tracker)
    high_t = tracker.args.track_high_thresh
    low_t = tracker.args.track_low_thresh
    print(f"BYTETracker: high={high_t} low={low_t} new={tracker.args.new_track_thresh} "
          f"buffer={tracker.args.track_buffer} match={tracker.args.match_thresh} "
          f"fuse_score={tracker.args.fuse_score}")
    if args.conf > low_t:
        print(f"note: detector conf {args.conf} > track_low_thresh {low_t} — the low-conf "
              f"second association will never fire (see docs/HOW_THE_TRACKER_WORKS.md §5)")

    writer = None
    rows = []
    prev_active: set[int] = set()
    seen_ids: set[int] = set()
    id_life: dict[int, int] = {}

    for fi, frame in enumerate(frames_from(args.source, args.seq)):
        if fi >= args.max_frames:
            break
        r = model.predict(frame, conf=min(args.conf, low_t) if args.conf <= low_t else args.conf,
                          device=args.device, verbose=False)[0]
        det = r.boxes.cpu().numpy()

        # Kalman-predicted positions BEFORE update (copy state, run predict math)
        pool = tracker.joint_stracks(
            [t for t in tracker.tracked_stracks if t.is_activated], tracker.lost_stracks)
        predicted = []
        for t in pool:
            if t.mean is None:
                continue
            m = t.mean.copy()
            m[:4] += m[4:]  # constant-velocity step (KalmanFilterXYAH motion model)
            cx, cy, a, h = m[:4]
            w = a * h
            predicted.append((t.track_id, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))

        n_high = int((det.conf >= high_t).sum()) if len(det) else 0
        n_low = int(((det.conf > low_t) & (det.conf < high_t)).sum()) if len(det) else 0

        tracks = tracker.update(det, frame)  # the real association happens here

        vis = frame.copy()
        for tid, box in predicted:
            draw_corners(vis, box, (0, 220, 255))
        if len(det):
            for b, cf in zip(det.xyxy, det.conf):
                x1, y1, x2, y2 = (int(v) for v in b)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 1)
                cv2.putText(vis, f"{cf:.2f}", (x1, max(10, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)

        active_now = set()
        for row in tracks:
            x1, y1, x2, y2, tid = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])
            active_now.add(tid)
            seen_ids.add(tid)
            id_life[tid] = id_life.get(tid, 0) + 1
            c = color_for(tid)
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            cv2.putText(vis, f"id:{tid}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)

        births = active_now - prev_active
        deaths = prev_active - active_now
        for k, tid in enumerate(sorted(births)):
            cv2.putText(vis, f"NEW id:{tid}", (10, 52 + 20 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2, cv2.LINE_AA)
        for k, tid in enumerate(sorted(deaths)):
            cv2.putText(vis, f"LOST id:{tid}", (vis.shape[1] - 130, 52 + 20 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(vis, f"f{fi}  det hi/lo {n_high}/{n_low}  active {len(active_now)}  "
                         f"lost {len(tracker.lost_stracks)}  ids so far {len(seen_ids)}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if writer is None:
            h_, w_ = vis.shape[:2]
            writer = cv2.VideoWriter(str(out / f"{stem}_anatomy.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 20, (w_, h_))
        writer.write(vis)

        rows.append({
            "frame": fi, "n_dets_high": n_high, "n_dets_low": n_low,
            "n_active": len(active_now), "n_lost": len(tracker.lost_stracks),
            "n_new": len(births), "n_removed_total": len(tracker.removed_stracks),
            "ids_active": " ".join(map(str, sorted(active_now))),
        })
        prev_active = active_now
        if fi % 30 == 0:
            print(f"  frame {fi}: dets {n_high}+{n_low}  active {len(active_now)}  "
                  f"lost {len(tracker.lost_stracks)}  born {sorted(births) or '-'}")

    if writer is not None:
        writer.release()
    if not rows:
        raise SystemExit("no frames processed")

    csv_path = out / f"{stem}_anatomy.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = [r["frame"] for r in rows]
    ax.plot(xs, [r["n_active"] for r in rows], label="active tracks", lw=1.6)
    ax.plot(xs, [r["n_lost"] for r in rows], label="lost (in buffer)", lw=1.2)
    ax.plot(xs, [r["n_dets_high"] for r in rows], label="high-conf dets", lw=1.0, alpha=0.7)
    ax.bar(xs, [r["n_new"] for r in rows], label="births", color="#d62828", alpha=0.5)
    cum = np.cumsum([r["n_new"] for r in rows])
    ax2 = ax.twinx()
    ax2.plot(xs, cum, color="gray", ls=":", label="cumulative IDs")
    ax2.set_ylabel("cumulative unique IDs")
    ax.set_xlabel("frame")
    ax.set_ylabel("count")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"ByteTrack internals over time — {stem}")
    fig.tight_layout()
    fig.savefig(out / f"{stem}_timeline.png", dpi=130)
    plt.close(fig)

    lives = list(id_life.values())
    print("\nSUMMARY")
    print(f"  frames               : {len(rows)}")
    print(f"  unique track IDs     : {len(seen_ids)}")
    print(f"  mean track length    : {np.mean(lives):.1f} frames" if lives else "  no tracks")
    print(f"  total births         : {int(sum(r['n_new'] for r in rows))}")
    print(f"  outputs -> {out}/{stem}_anatomy.mp4, {csv_path.name}, {stem}_timeline.png")


if __name__ == "__main__":
    main()
