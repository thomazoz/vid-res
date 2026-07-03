"""Follow ONE object through a multi-object tracker's output.

ByteTrack/BoT-SORT track everything and recycle IDs when detections drop out
(on OTB sequences the followed ID often dies within ~10 frames — see
docs/HOW_THE_TRACKER_WORKS.md). `TargetFollower` turns that stream into a
single-target tracker for visualization: it locks onto one object and, when
its track ID disappears, re-acquires the best-overlapping new track instead of
going dark. The number of ID switches is counted and reported.

Seeding: a reference box (e.g. OTB ground truth) if provided, else the largest
box in the first frame with detections, else an explicit track ID.

Usage:
    from single_target import TargetFollower
    follower = TargetFollower()
    box, tid, status = follower.update(ids, xyxy, ref_box=gt_xyxy)
    # status: "tracked" | "acquired" | "reacquired" | "lost" | "waiting"
"""

from __future__ import annotations

import numpy as np


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


class TargetFollower:
    """Lock onto one track ID; re-acquire by overlap when the ID vanishes."""

    def __init__(self, reacquire_iou: float = 0.15, want_id: int | None = None):
        self.reacquire_iou = reacquire_iou
        self.want_id = want_id          # explicit ID to acquire first (optional)
        self.target_id: int | None = None
        self.last_box: np.ndarray | None = None
        self.target_cls: int | None = None
        self.n_switches = 0             # re-acquisitions after the first lock
        self.frames_tracked = 0
        self.frames_lost = 0

    def _best_overlap(self, boxes, ref_box, ids, classes):
        """Index of the box best overlapping ref_box (>= reacquire_iou), same
        class preferred when known."""
        best_i, best_v = None, self.reacquire_iou
        for i, b in enumerate(boxes):
            v = iou_xyxy(b, ref_box)
            if classes is not None and self.target_cls is not None \
                    and int(classes[i]) == self.target_cls:
                v *= 1.25               # prefer keeping the same class
            if v > best_v:
                best_v, best_i = v, i
        return best_i

    def update(self, ids, boxes, classes=None, ref_box=None):
        """Feed one frame of tracker output.

        Args:
            ids: iterable of track IDs (ints) for this frame.
            boxes: matching xyxy boxes (array-like, shape [n, 4]).
            classes: optional matching class indices.
            ref_box: optional reference xyxy (e.g. ground truth) used to seed
                and to re-acquire; falls back to the last known box.

        Returns:
            (box, track_id, status) — box is None while nothing is locked.
        """
        ids = [int(i) for i in ids]
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)

        def lock(i, status):
            first = self.target_id is None
            if not first and ids[i] != self.target_id:
                self.n_switches += 1
            self.target_id = ids[i]
            self.last_box = boxes[i].copy()
            if classes is not None:
                self.target_cls = int(classes[i])
            self.frames_tracked += 1
            return boxes[i], self.target_id, status

        # first acquisition
        if self.target_id is None:
            if self.want_id is not None and self.want_id in ids:
                return lock(ids.index(self.want_id), "acquired")
            if self.want_id is None and len(boxes):
                if ref_box is not None:
                    i = self._best_overlap(boxes, ref_box, ids, classes)
                    if i is not None:
                        return lock(i, "acquired")
                else:  # largest box
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    return lock(int(np.argmax(areas)), "acquired")
            return None, None, "waiting"

        # target still present
        if self.target_id in ids:
            return lock(ids.index(self.target_id), "tracked")

        # target ID vanished — try to re-acquire near the reference / last box
        ref = ref_box if ref_box is not None else self.last_box
        if len(boxes) and ref is not None:
            i = self._best_overlap(boxes, ref, ids, classes)
            if i is not None:
                return lock(i, "reacquired")

        self.frames_lost += 1
        return None, self.target_id, "lost"

    def summary(self) -> str:
        total = self.frames_tracked + self.frames_lost
        pct = 100.0 * self.frames_tracked / total if total else 0.0
        return (f"target followed {self.frames_tracked}/{total} frames ({pct:.0f}%), "
                f"{self.n_switches} ID switch(es), lost {self.frames_lost} frames")
