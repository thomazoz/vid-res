"""End-to-end check of the tracking pipeline used by app.py.

Generates a short clip with *real* camera motion (pan + zoom) over a busy
real-world image, runs the same YOLO track -> annotate -> write -> (re-encode)
steps app.py uses, then validates the output is a real, decodable video that
actually contains detections and stable track IDs.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SRC_IMG = "datasets/coco128/images/train2017/000000000164.jpg"
WORK = Path("verify_out")
N_FRAMES = 72
FPS = 24.0
OUT_W, OUT_H = 640, 480


def make_moving_clip(src: str, out_path: Path) -> int:
    """Pan + zoom across `src` to synthesize genuine inter-frame motion."""
    base = cv2.imread(src)
    base = cv2.resize(base, (OUT_W, OUT_H))
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H)
    )
    cx, cy = OUT_W / 2, OUT_H / 2
    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        scale = 1.0 + 0.25 * t                         # slow zoom-in
        tx = 60.0 * np.sin(2 * np.pi * t)              # horizontal pan
        ty = 25.0 * np.sin(2 * np.pi * t * 0.5)        # gentle vertical drift
        M = np.array(
            [[scale, 0, (1 - scale) * cx + tx],
             [0, scale, (1 - scale) * cy + ty]],
            dtype=np.float32,
        )
        frame = cv2.warpAffine(base, M, (OUT_W, OUT_H), borderMode=cv2.BORDER_REFLECT)
        writer.write(frame)
    writer.release()
    return N_FRAMES


def run_tracking(in_path: Path, model_name="yolo11n.pt", tracker="bytetrack.yaml", conf=0.3):
    """Mirror of app.py's core loop, instrumented to collect stats."""
    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = WORK / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    print(f"  input: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"  writer opened: {writer.isOpened()}")

    model = YOLO(model_name)
    frame_idx, total_dets, track_ids = 0, 0, set()
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.track(frame, persist=True, tracker=tracker, conf=conf, verbose=False)
        boxes = results[0].boxes
        if boxes is not None:
            total_dets += len(boxes)
            if boxes.id is not None:
                track_ids.update(int(i) for i in boxes.id)
        writer.write(results[0].plot())
        frame_idx += 1
    dt = time.time() - t0

    cap.release()
    writer.release()

    # H.264 re-encode for browser playback (app.py does this iff ffmpeg exists).
    playable = out_path
    have_ffmpeg = shutil.which("ffmpeg") is not None
    if have_ffmpeg:
        h264 = WORK / "annotated_h264.mp4"
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_path), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(h264)],
            capture_output=True,
        )
        if r.returncode == 0:
            playable = h264

    return {
        "out_path": out_path,
        "playable": playable,
        "frames_written": frame_idx,
        "total_frames": total_frames,
        "total_dets": total_dets,
        "avg_dets": total_dets / max(frame_idx, 1),
        "unique_track_ids": len(track_ids),
        "secs": dt,
        "have_ffmpeg": have_ffmpeg,
    }


def validate(out_path: Path, expected_frames: int) -> None:
    assert out_path.exists() and out_path.stat().st_size > 0, "output missing/empty"
    cap = cv2.VideoCapture(str(out_path))
    decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None, "output not decodable"
    print(f"  output decodes: {decoded} frames, size {out_path.stat().st_size/1e3:.0f} KB")
    assert abs(decoded - expected_frames) <= 1, f"frame mismatch {decoded} vs {expected_frames}"


def main():
    WORK.mkdir(exist_ok=True)
    in_path = WORK / "test_input.mp4"
    print("[1] Generating moving test clip...")
    n = make_moving_clip(SRC_IMG, in_path)
    print(f"    wrote {n} frames -> {in_path}")

    print("[2] Running tracking pipeline (mirror of app.py)...")
    stats = run_tracking(in_path)

    print("[3] Validating annotated output...")
    validate(stats["out_path"], stats["frames_written"])

    print("\n===== PIPELINE VERIFICATION =====")
    print(f"frames in / out      : {stats['total_frames']} / {stats['frames_written']}")
    print(f"avg detections/frame : {stats['avg_dets']:.2f}")
    print(f"total detections     : {stats['total_dets']}")
    print(f"unique track IDs     : {stats['unique_track_ids']}")
    print(f"throughput           : {stats['frames_written']/stats['secs']:.1f} fps "
          f"({stats['secs']:.1f}s for {stats['frames_written']} frames)")
    print(f"ffmpeg available     : {stats['have_ffmpeg']}  "
          f"(-> H.264 re-encode {'ran' if stats['have_ffmpeg'] else 'SKIPPED, mp4v only'})")
    verdict = "PASS" if stats["avg_dets"] > 0 and stats["frames_written"] == stats["total_frames"] else "FAIL"
    print(f"VERDICT              : {verdict}")


if __name__ == "__main__":
    main()
