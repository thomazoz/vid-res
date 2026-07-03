#!/usr/bin/env python3
"""
camera_motion.py — Measure camera motion speed in a video.

Approach
--------
1. Read video frame-by-frame with cv2.VideoCapture.
2. Detect Shi-Tomasi good features in the previous frame and track them
   into the current frame with Lucas-Kanade sparse optical flow
   (cv2.calcOpticalFlowPyrLK).
3. Reject foreground / independently-moving tracks with a RANSAC-fitted
   affine model (cv2.estimateAffine2D). Surviving inliers are treated
   as the static background whose motion is dominated by the camera.
4. From the affine matrix we derive the camera motion components:
       tx, ty  — translation (px/frame)
       angle   — rotation (rad/frame)
       scale   — zoom / forward motion factor (1 == none, >1 == zoom-in)
5. Convert per-frame quantities to real-world units when known:
       px/sec  =  magnitude(tx, ty) * fps
       deg/sec =  abs(angle) * 180/pi * fps
       zoom/sec=  log(scale) * fps       (signed: + = in, - = out)
   Approximate ground speed can be estimated from a horizon line and a
   pitch angle (--pitch-deg) — useful for drone / driving footage.
6. Emit per-frame CSV, a video with drawn tracks, and a Matplotlib plot
   of the four signals.

Usage
-----
    python camera_motion.py INPUT_VIDEO [--out-dir OUT] [options]

    python camera_motion.py clip.mp4
    python camera_motion.py clip.mp4 --out-dir ./cam_out --save-video
    python camera_motion.py dashcam.mp4 --horizon-y 380 --pitch-deg 6
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------- optional matplotlib (only required for the summary plot) -----
try:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    _HAVE_PLT = True
except ImportError:  # pragma: no cover
    _HAVE_PLT = False


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class FrameMotion:
    """Camera motion between frame N and frame N+1."""

    frame: int            # index of the *current* frame (0-based)
    timestamp: float      # seconds
    tx: float             # translation x (px)
    ty: float             # translation y (px)
    angle_deg: float      # rotation (deg)
    scale: float          # zoom factor
    inliers: int          # number of RANSAC inliers
    p0: int               # features detected
    px_per_sec: float     # translation magnitude * fps
    deg_per_sec: float    # rotation magnitude * fps (always >= 0)
    zoom_per_sec: float   # log(scale) * fps (signed)


# --------------------------------------------------------------------------
# Core tracker
# --------------------------------------------------------------------------
class CameraMotionEstimator:
    """Sparse Lucas-Kanade tracker with RANSAC-based moving-object rejection."""

    def __init__(
        self,
        max_corners: int = 800,
        quality_level: float = 0.01,
        min_distance: int = 12,
        block_size: int = 7,
        win_size: tuple[int, int] = (21, 21),
        max_level: int = 3,
        ransac_thresh: float = 3.0,
    ) -> None:
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.ransac_thresh = ransac_thresh

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _grayscale(frame: np.ndarray) -> np.ndarray:
        return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _detect(self, gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        return pts.reshape(-1, 1, 2) if pts is not None else np.empty((0, 1, 2), np.float32)

    def _track(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray, prev_pts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (curr_pts, status, err). Status=1 means track kept."""
        if prev_pts.size == 0:
            empty = np.empty((0, 1, 2), np.float32)
            return empty, np.empty((0, 1), np.uint8), np.empty((0, 1), np.float32)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )
        if curr_pts is None:
            return (
                np.empty((0, 1, 2), np.float32),
                np.empty((0, 1), np.uint8),
                np.empty((0, 1), np.float32),
            )
        return curr_pts, status, err

    # ---- per-frame step --------------------------------------------------
    def step(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Run one tracking step.

        Returns
        -------
        affine : (2, 3) float64 — RANSAC inlier affine, or None if <3 inliers.
        inlier_mask : (N, 1) uint8 — 1 where the track is a static inlier.
        new_pts : (N, 1, 2) float32 — points to track in the *next* frame.
        """
        if prev_pts is None or prev_pts.size == 0:
            prev_pts = self._detect(prev_gray)
        curr_pts, status, _ = self._track(prev_gray, curr_gray, prev_pts)
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]
        if good_prev.shape[0] < 3:
            return (
                np.empty((0,)),  # sentinel: caller checks
                np.zeros((good_prev.shape[0], 1), np.uint8),
                self._detect(curr_gray),
            )

        affine, inlier_mask = cv2.estimateAffine2D(
            good_prev,
            good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=2000,
            confidence=0.999,
        )
        if affine is None:
            return (
                np.empty((0,)),
                np.zeros((good_prev.shape[0], 1), np.uint8),
                self._detect(curr_gray),
            )

        inlier_mask = inlier_mask.reshape(-1, 1).astype(np.uint8)
        # Re-detect on every Nth frame would be more elaborate; for now
        # re-detect whenever inliers fall below half the feature budget.
        new_pts = (
            good_curr[inlier_mask.flatten() == 1].reshape(-1, 1, 2)
            if inlier_mask.sum() >= self.feature_params["maxCorners"] // 2
            else self._detect(curr_gray)
        )
        return affine, inlier_mask, new_pts


# --------------------------------------------------------------------------
# Affine decomposition
# --------------------------------------------------------------------------
def decompose_affine(M: np.ndarray) -> tuple[float, float, float, float]:
    """
    Decompose a 2x3 affine into (tx, ty, angle_rad, scale).
    Assumes the linear part is a uniform similarity
    (rotation + uniform scale, no shear / non-uniform aspect).
    """
    A = M[:, :2]
    t = M[:, 2]
    # singular values: sqrt(lam) of A^T A
    sx = math.hypot(A[0, 0], A[0, 1])
    sy = math.hypot(A[1, 0], A[1, 1])
    # For a rotation+scale: A = s * R. We pick the average scale and use
    # the angle from the first row (atan2 handles quadrant).
    scale = 0.5 * (sx + sy)
    if scale < 1e-9:
        angle = 0.0
    else:
        angle = math.atan2(A[0, 1], A[0, 0])
    return float(t[0]), float(t[1]), float(angle), float(scale)


# --------------------------------------------------------------------------
# Speed / distance helpers
# --------------------------------------------------------------------------
def forward_speed_px_per_sec(
    ty_px: float, fps: float, horizon_y: Optional[int], pitch_deg: Optional[float]
) -> Optional[float]:
    """
    Convert a vertical pixel translation into an approximate
    ground-plane speed (px/sec) using a pinhole + flat-ground model.

    Derivation
    ----------
    With camera height H, focal length f (px), pitch theta below horizon,
    and a flat ground plane, a stationary ground point at depth Z
    projects to y = (H * f) / Z on the image (with y increasing *up*).
    In image coords (y increasing *down*) the projection is
        y_img = horizon_y + (H * f) / Z
    The relationship between vertical flow v (px/frame) at row y and
    camera forward speed V (units/frame, V = V_real / fps) is
        v = - V * (y_img - horizon_y) / (H * f / (y_img - horizon_y) + V)
    For points near the horizon (V << f * H / (y - horizon_y)) the
    far-field approximation gives v ≈ -V * (y - horizon_y)^2 / (H * f).
    Rearranged: V ≈ -v * H * f / (y - horizon_y)^2.

    The script does not know H or f. We return a *normalized* speed in
    px/sec equivalent units — multiply by (H * f) to get true world
    units. With pitch_deg given, the focal length f cancels in the
    pitch-only ratio below, and we return the relative speed scaled
    by tan(pitch) so the caller can convert with a single scale factor.
    """
    if horizon_y is None or pitch_deg is None or abs(pitch_deg) < 1e-3:
        return None
    return float(ty_px) * fps * math.tan(math.radians(pitch_deg))


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def process_video(
    video_path: str,
    out_dir: str,
    save_video: bool = False,
    horizon_y: Optional[int] = None,
    pitch_deg: Optional[float] = None,
    progress_every: int = 30,
) -> list[FrameMotion]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(out_dir, "camera_motion.csv")
    video_out_path = os.path.join(out_dir, "camera_motion_overlay.mp4")

    writer: Optional[cv2.VideoWriter] = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))

    est = CameraMotionEstimator()
    ret, prev = cap.read()
    if not ret:
        cap.release()
        raise SystemExit("Video has no readable frames.")
    prev_gray = est._grayscale(prev)
    prev_pts = est._detect(prev_gray)

    motions: list[FrameMotion] = []
    frame_idx = 0
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "frame",
            "timestamp_s",
            "tx_px",
            "ty_px",
            "angle_deg",
            "scale",
            "inliers",
            "px_per_sec",
            "deg_per_sec",
            "zoom_per_sec",
        ]
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        curr_gray = est._grayscale(frame)
        affine, inlier_mask, prev_pts = est.step(prev_gray, curr_gray, prev_pts)
        timestamp = frame_idx / fps

        if affine is None or affine.size == 0:
            mot = FrameMotion(
                frame=frame_idx,
                timestamp=timestamp,
                tx=0.0,
                ty=0.0,
                angle_deg=0.0,
                scale=1.0,
                inliers=0,
                p0=int(prev_pts.shape[0]),
                px_per_sec=0.0,
                deg_per_sec=0.0,
                zoom_per_sec=0.0,
            )
        else:
            tx, ty, ang_rad, scale = decompose_affine(affine)
            inliers = int(inlier_mask.sum())
            px_per_sec = math.hypot(tx, ty) * fps
            deg_per_sec = abs(ang_rad) * (180.0 / math.pi) * fps
            zoom_per_sec = (math.log(scale) if scale > 0 else 0.0) * fps
            mot = FrameMotion(
                frame=frame_idx,
                timestamp=timestamp,
                tx=tx,
                ty=ty,
                angle_deg=math.degrees(ang_rad),
                scale=scale,
                inliers=inliers,
                p0=int(prev_pts.shape[0]),
                px_per_sec=px_per_sec,
                deg_per_sec=deg_per_sec,
                zoom_per_sec=zoom_per_sec,
            )
        motions.append(mot)
        csv_writer.writerow(
            [
                mot.frame,
                f"{mot.timestamp:.4f}",
                f"{mot.tx:.3f}",
                f"{mot.ty:.3f}",
                f"{mot.angle_deg:.4f}",
                f"{mot.scale:.5f}",
                mot.inliers,
                f"{mot.px_per_sec:.3f}",
                f"{mot.deg_per_sec:.4f}",
                f"{mot.zoom_per_sec:.5f}",
            ]
        )

        if writer is not None:
            overlay = frame.copy()
            label = (
                f"px/s={mot.px_per_sec:6.1f}  "
                f"deg/s={mot.deg_per_sec:5.2f}  "
                f"zoom/s={mot.zoom_per_sec:+.3f}  "
                f"inliers={mot.inliers:3d}/{mot.p0:3d}"
            )
            cv2.putText(
                overlay,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(overlay)

        if progress_every and frame_idx % progress_every == 0 and total:
            pct = 100.0 * frame_idx / total
            print(
                f"\r  frame {frame_idx:>6d}/{total}  ({pct:5.1f}%)  "
                f"px/s={mot.px_per_sec:6.1f}  inliers={mot.inliers:3d}",
                end="",
                flush=True,
            )

        prev_gray = curr_gray

    cap.release()
    csv_file.close()
    if writer is not None:
        writer.release()
    if progress_every:
        print()

    if motions:
        avg_px = float(np.mean([m.px_per_sec for m in motions]))
        max_px = float(np.max([m.px_per_sec for m in motions]))
        avg_deg = float(np.mean([m.deg_per_sec for m in motions]))
        avg_zoom = float(np.mean([m.zoom_per_sec for m in motions]))
        print(
            f"\nSummary [{video_path}]  {frame_idx} frames @ {fps:.2f} fps, "
            f"{w}x{h}"
        )
        print(
            f"  translation : mean={avg_px:7.2f} px/s   peak={max_px:7.2f} px/s"
        )
        print(f"  rotation    : mean={avg_deg:7.3f} deg/s")
        print(f"  zoom rate   : mean={avg_zoom:+7.4f} /s "
              f"({'in' if avg_zoom > 0 else 'out'})")
        if horizon_y is not None and pitch_deg is not None:
            print(
                f"  pitch={pitch_deg:.2f}°, horizon_y={horizon_y}  → "
                f"relative forward speed = "
                f"{forward_speed_px_per_sec(np.mean([m.ty for m in motions]), fps, horizon_y, pitch_deg):.3f}"
                f" (px/s)·tan(pitch)"
            )

    return motions


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def plot_motion(motions: list[FrameMotion], out_path: str) -> None:
    if not _HAVE_PLT or not motions:
        if not _HAVE_PLT:
            print("  (matplotlib not installed; skipping plot)")
        return
    t = [m.timestamp for m in motions]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, [m.px_per_sec for m in motions], lw=1.2)
    axes[0].set_ylabel("translation\npx / s")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, [m.deg_per_sec for m in motions], lw=1.2, color="tab:orange")
    axes[1].set_ylabel("rotation\ndeg / s")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, [m.zoom_per_sec for m in motions], lw=1.2, color="tab:green")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("zoom rate\nlog(s)/s")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Camera motion over time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  plot saved → {out_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure camera motion speed in a video.",
    )
    p.add_argument("video", help="Path to input video")
    p.add_argument(
        "--out-dir",
        default="camera_motion_out",
        help="Directory to write CSV / plot / overlay (default: camera_motion_out)",
    )
    p.add_argument(
        "--save-video",
        action="store_true",
        help="Write an MP4 with per-frame overlays (slower).",
    )
    p.add_argument(
        "--horizon-y",
        type=int,
        default=None,
        help="Optional y-coordinate of the horizon line (px from top).",
    )
    p.add_argument(
        "--pitch-deg",
        type=float,
        default=None,
        help="Optional camera pitch below horizon in degrees.",
    )
    p.add_argument(
        "--max-corners",
        type=int,
        default=800,
        help="Max Shi-Tomasi corners to track (default: 800).",
    )
    p.add_argument(
        "--quality-level",
        type=float,
        default=0.01,
        help="Shi-Tomasi quality level (default: 0.01).",
    )
    p.add_argument(
        "--min-distance",
        type=int,
        default=12,
        help="Min distance between features (default: 12).",
    )
    p.add_argument(
        "--ransac-thresh",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold in px (default: 3.0).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the matplotlib summary plot.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if not os.path.isfile(args.video):
        print(f"Error: '{args.video}' is not a file.", file=sys.stderr)
        return 2

    motions = process_video(
        video_path=args.video,
        out_dir=args.out_dir,
        save_video=args.save_video,
        horizon_y=args.horizon_y,
        pitch_deg=args.pitch_deg,
    )

    if not args.no_plot:
        plot_motion(motions, os.path.join(args.out_dir, "camera_motion.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
