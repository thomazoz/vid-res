# How the tracker works

This project tracks objects with `model.track(frame, persist=True)` from
Ultralytics 8.4.61, using the **ByteTrack** algorithm by default
(`bytetrack.yaml`) or **BoT-SORT** (`botsort.yaml`). This document explains the
exact mechanism, grounded in the installed source at
`/opt/anaconda3/lib/python3.13/site-packages/ultralytics/trackers/`
(files cited throughout). Every number comes from that source or its config
files — nothing is from memory.

**The one-paragraph version:** YOLO only *detects* — each frame it returns
boxes with no notion of identity. The tracker adds identity: it keeps a set of
*tracks* (one per object, each with a Kalman-filter motion model), predicts
where each track's box should be this frame, and then solves an assignment
problem matching predicted boxes to the new detections by overlap. Matched
tracks keep their ID; unmatched detections may start new IDs; unmatched tracks
are kept as "lost" for a grace period (30 frames by default) before being
removed. ByteTrack's twist is that it runs this matching **twice** — first with
confident detections, then with low-confidence ones — so an object whose
confidence dips (blur! occlusion!) can still hold onto its ID.

---

## 1. Where tracking hooks into YOLO (`track.py`)

`model.track()` is `model.predict()` plus two callbacks registered by
`register_tracker()` (`track.py:103`):

- **`on_predict_start`** (`track.py:18`): loads the tracker yaml, builds a
  `BYTETracker` or `BOTSORT` instance per video stream. With `persist=True`
  and an existing tracker, it returns early — **this is what keeps IDs alive
  across your frame-by-frame loop** (`track.py:33-34`). Without persist, a new
  video path triggers `tracker.reset()` (`track.py:88-90`), zeroing the ID
  counter.
- **`on_predict_postprocess_end`** (`track.py:71`): after YOLO produces boxes,
  it calls `tracker.update(det, img, feats)` and **overwrites the frame's
  results with the tracker's output** — only detections that were matched to an
  active track survive, now carrying a `track_id` (`results[0].boxes.id`).

So the flow in `app.py` / `run_algo.py` per frame is:
`YOLO inference → NMS → tracker.update() → annotated result with IDs`.

## 2. The state each track carries (`byte_tracker.py`, `kalman_filter.py`)

Each track is an `STrack` (`byte_tracker.py:16`) holding an **8-dimensional
Kalman filter state**: `(x, y, a, h, vx, vy, va, vh)` — box center, aspect
ratio, height, and their velocities (`KalmanFilterXYAH`,
`kalman_filter.py:6-13`). Motion model: **constant velocity** — the state
transition just adds each velocity to its position once per frame
(`kalman_filter.py.__init__`, dt = 1.0). Uncertainty is scaled relative to box
height (`_std_weight_position = 1/20`, `_std_weight_velocity = 1/160`).

- `predict()` moves the box to where the object *should* be this frame.
  For non-tracked (lost) tracks the aspect-ratio velocity is zeroed first
  (`byte_tracker.py:82`).
- `update()` (a Kalman correction) blends the prediction with the matched
  detection box (`byte_tracker.py:148`).
- BoT-SORT's `BOTrack` subclasses this with a `KalmanFilterXYWH` variant —
  state `(x, y, w, h, vx, vy, vw, vh)` — i.e. it tracks width directly instead
  of aspect ratio (`bot_sort.py:21,89-94`).

Track lifecycle states (`basetrack.py`): **New → Tracked → Lost → Removed**.

## 3. One frame through `BYTETracker.update()` (`byte_tracker.py:282`)

```
                         YOLO detections (after NMS, at your --conf)
                                        │
             ┌──────────────────────────┼─────────────────────────┐
             │ conf ≥ track_high_thresh │ track_low_thresh < conf │  conf ≤ track_low_thresh
             │        (0.25)            │      < 0.25             │       (0.1) → discarded
             ▼                          ▼                         
        HIGH bucket                LOW bucket                     
             │                          │
   [Kalman predict all active+lost tracks]        (byte_tracker.py:315)
   [BoT-SORT only: GMC camera-motion warp]        (byte_tracker.py:316-323)
             │                          │
   ── ASSOCIATION ROUND 1 ──────────────┼──────────────────────────
   cost = 1 − IoU(predicted box, det box), optionally × det score
   solved by lap.lapjv Hungarian solver, gate: match_thresh 0.8
             │                          │
   matched → track.update() keeps ID    │
   unmatched tracks ────────────────────┤
                                        ▼
   ── ASSOCIATION ROUND 2 (the "Byte" trick) ─────────────────────
   remaining *Tracked* tracks vs LOW bucket, pure IoU, gate 0.5
   matched → ID survives a low-confidence frame (blur/occlusion!)
   still unmatched tracks → mark_lost()
                                        │
   ── UNCONFIRMED ROUND ── new-born (1-frame-old) tracks vs leftover
   high dets, gate 0.7; unmatched newborns are killed immediately
                                        │
   ── BIRTH ── leftover high detections with conf ≥ new_track_thresh
   (0.25) become new tracks with fresh IDs        (byte_tracker.py:370-376)
                                        │
   ── REAP ── lost tracks older than track_buffer (30 frames)
   are Removed permanently               (byte_tracker.py:377-381)
```

Key mechanics, per the source:

- **Cost function.** Round 1 cost is IoU distance (`1 − IoU`) between each
  track's *Kalman-predicted* box and each detection (`matching.iou_distance`).
  With `fuse_score: True` (default in both yamls) the IoU similarity is
  multiplied by the detection confidence (`matching.fuse_score`,
  `matching.py:140-162`) — a weak detection must overlap *better* to win a
  match.
- **The assignment** is a global optimum, not greedy: `lap.lapjv` (Jonker–
  Volgenant, the reason this project needs the `lap` package) with
  `cost_limit = match_thresh` (`matching.py:44`).
- **Second association** (`byte_tracker.py:337-352`) is what makes ByteTrack
  "Byte": detections with conf between `track_low_thresh` (0.1) and
  `track_high_thresh` (0.25) — which most trackers throw away — get a second
  matching round against the tracks that round 1 left unmatched, pure IoU with
  a fixed 0.5 gate. An object that got blurry for a few frames usually still
  produces a low-confidence box, so its track survives instead of dying.
- **Re-activation** (`byte_tracker.py:132`): a *Lost* track matched in either
  round is re-activated **with its original ID** — this is how identity
  survives short occlusions.
- **De-duplication**: tracked and lost lists are cross-checked; pairs with IoU
  distance < 0.15 keep only the longer-lived copy (`byte_tracker.py:455-469`).

## 4. What BoT-SORT adds (`bot_sort.py`)

BoT-SORT is ByteTrack plus two upgrades (same two-round association skeleton):

1. **Global Motion Compensation (GMC).** Before matching, it estimates the
   *camera's* motion between frames — default method `sparseOptFlow`
   (`botsort.yaml`), i.e. the same Lucas-Kanade sparse optical flow idea as
   this project's `camera_motion.py` — and warps every track's predicted box by
   that transform (`STrack.multi_gmc`, `byte_tracker.py:100-117`, applied at
   `byte_tracker.py:316-323`). This keeps predictions aligned when the camera
   pans/shakes.
2. **Appearance (ReID) fusion** — **off by default** (`with_reid: False` in
   `botsort.yaml`). When enabled, the cost becomes
   `min(IoU-based cost, embedding cost)` where the embedding cost is cosine
   distance halved, hard-gated: candidates with appearance distance above
   `1 − appearance_thresh` (0.8) or IoU proximity below `proximity_thresh`
   (0.5) are excluded (`bot_sort.py:210-223`). This lets a re-appearing object
   be re-identified by *how it looks*, not just where it is.

## 5. Every tracker parameter and what it does

| Parameter (yaml) | Default | Meaning | Effect of changing it |
|---|---|---|---|
| `tracker_type` | bytetrack / botsort | which algorithm | botsort = + camera compensation, optional ReID, ~slower |
| `track_high_thresh` | 0.25 | min conf for round-1 (high) bucket | raise → cleaner tracks, more misses; lower → more matches, more noise |
| `track_low_thresh` | 0.1 | floor for round-2 (low) bucket | lower → recovers weaker detections but risks drift onto noise |
| `new_track_thresh` | 0.25 | min conf to *birth* a new ID | raise → fewer phantom IDs; lower → catch faint new objects |
| `track_buffer` | 30 | frames a lost track is kept before removal | raise → survives longer occlusions, but stale tracks may steal IDs |
| `match_thresh` | 0.8 | lap.lapjv cost gate in round 1 | raise → more permissive matching; lower → stricter, more fragmentation |
| `fuse_score` | True | multiply IoU similarity by det confidence | off → position-only matching |
| `gmc_method` (BoT-SORT) | sparseOptFlow | camera-motion estimator | `orb`/`sift`/`ecc` alternatives; `none` disables |
| `proximity_thresh` (BoT-SORT) | 0.5 | min IoU for a ReID match to be considered | raise → ReID only for near-overlapping candidates |
| `appearance_thresh` (BoT-SORT) | 0.8 | min appearance similarity for ReID | raise → fewer but safer identity revivals |
| `with_reid` (BoT-SORT) | False | enable appearance embeddings | on → better through-occlusion identity, slower |

Note the interaction with the detector's own `conf` argument: our scripts run
`model.track(conf=0.25–0.3)`, which filters detections *before* the tracker
sees them — so the "low bucket" (0.1–0.25) only exists if `conf` is set at or
below `track_low_thresh`… in practice with `conf=0.3` **ByteTrack's second
association never fires** because nothing below 0.3 survives to reach it. Run
with `conf=0.1` if you want the full Byte behavior (the tracker's own
thresholds then do the filtering).

## 6. Why this explains our experimental findings

- **Blur kills recall → tracks die in ~`track_buffer` frames.** Under motion
  blur the detector goes silent (our studies: recall collapses while precision
  holds). No detection ⇒ no match ⇒ `mark_lost()`; after 30 frames the track
  is Removed. When the object re-sharpens it gets a **new ID** — this is the
  ID-churn we measured (e.g. 45 unique IDs in 200 frames of Basketball for
  ~10 visible objects).
- **The low-confidence second pass is the built-in blur defense** — but it
  only works if detector `conf` ≤ `track_low_thresh` (see note above). This is
  a concrete, testable tuning lever for the blur experiments.
- **Fast/erratic motion breaks the constant-velocity assumption.** The Kalman
  prediction lands where the object *would* be at constant velocity; a
  direction change (Basketball, Bolt) leaves the prediction with low IoU
  against the true box, failing the `match_thresh` gate → fragmentation. This
  is why our case analysis finds fast_motion sequences failing.
- **Camera motion moves *every* box** — ByteTrack has no compensation, so a
  panning camera degrades all predictions at once; BoT-SORT's GMC fixes
  exactly this. Prefer **BoT-SORT when the camera moves or scenes are
  crowded**; ByteTrack is leaner for fixed-camera footage.
- **Small objects die first** twice over: blur erases their detections
  (measured: 14% retention at 25 px blur), and their small boxes make IoU
  matching brittle — a few pixels of prediction error is a large IoU drop for
  a small box.

## 7. See it live

`tracker_anatomy.py` (this project) runs the real `BYTETracker` against YOLO
detections and renders its internals per frame: green = raw detections,
yellow = each track's Kalman-predicted box *before* it sees the detections,
solid boxes with IDs = tracks after update, plus birth/death flashes and a
timeline of active/lost/new/removed counts:

```bash
python3 tracker_anatomy.py samples/people.mp4 --max-frames 150
python3 tracker_anatomy.py --seq Basketball --max-frames 200   # OTB sequence
```

Outputs land in `tracker_anatomy_out/` (annotated MP4, per-frame CSV,
timeline PNG).
