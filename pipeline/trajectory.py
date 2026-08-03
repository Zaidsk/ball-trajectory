"""
Generalized trajectory builder.

Takes a list of confirmed anchor points (frame_idx, x, y) -- however
they were obtained, whether from the automatic Kalman-gated tracker
(clean footage) or human-confirmed clicks (hard broadcast footage) --
and produces:
  1. A dense per-frame interpolated/extrapolated (x, y) track
  2. Segment-wise physics metrics (pixel-space speed & direction) between
     each consecutive pair of anchors, so direction/speed changes (e.g.
     bat contact) are visible rather than averaged away
  3. A rendered, annotated output video

This intentionally does not fabricate real-world units (mph, meters).
Without camera calibration and a known slow-motion factor for the
source footage, only pixel-space and relative metrics are honest to
report -- see README for why.
"""
import cv2
import numpy as np
import os


def build_track(anchors, extrapolate_to=None):
    """
    anchors: list of (frame_idx, x, y), sorted by frame_idx, len >= 2.
    extrapolate_to: optional frame_idx to extrapolate beyond the last
        anchor using its segment's velocity. If None, track ends at the
        last anchor.

    Returns dict: frame_idx -> (x, y, is_confirmed_anchor)
    """
    anchors = sorted(anchors, key=lambda a: a[0])
    if len(anchors) < 2:
        raise ValueError("Need at least 2 anchors to interpolate a trajectory")

    track = {}
    for i in range(len(anchors) - 1):
        f0, x0, y0 = anchors[i]
        f1, x1, y1 = anchors[i + 1]
        for f in range(f0, f1 + 1):
            t = (f - f0) / (f1 - f0)
            track[f] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0), f in {a[0] for a in anchors})

    if extrapolate_to is not None and extrapolate_to > anchors[-1][0]:
        f_prev, x0, y0 = anchors[-2]
        f_last, x1, y1 = anchors[-1]
        vx = (x1 - x0) / (f_last - f_prev)
        vy = (y1 - y0) / (f_last - f_prev)
        for f in range(f_last + 1, extrapolate_to + 1):
            dt = f - f_last
            track[f] = (x1 + vx * dt, y1 + vy * dt, False)

    return track


def segment_physics(anchors):
    """Returns list of dicts describing speed/direction between each consecutive anchor pair."""
    anchors = sorted(anchors, key=lambda a: a[0])
    results = []
    for i in range(len(anchors) - 1):
        f0, x0, y0 = anchors[i]
        f1, x1, y1 = anchors[i + 1]
        dx, dy, df = x1 - x0, y1 - y0, f1 - f0
        speed = float(np.hypot(dx, dy) / df)
        angle = float(np.degrees(np.arctan2(dy, dx)))
        results.append({
            "from_frame": f0, "to_frame": f1,
            "speed_px_per_frame": round(speed, 2),
            "direction_deg": round(angle, 1),
        })
    return results


def render_video(frames_dir, track, out_path, fps=25, slow_factor=4, trail_len=20):
    """Render the track overlaid on the source frames and export as mp4."""
    frame_indices = sorted(track.keys())
    tmp_dir = out_path + "_frames_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    trail = []
    for idx in frame_indices:
        frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        x, y, is_anchor = track[idx]
        trail.append((x, y))
        if len(trail) > trail_len:
            trail.pop(0)

        overlay = frame.copy()
        for (tx, ty) in trail:
            cv2.circle(overlay, (int(tx), int(ty)), 4, (60, 200, 255), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        color = (0, 230, 0) if is_anchor else (0, 0, 230)
        cv2.circle(frame, (int(x), int(y)), 10, color, 2)

        cv2.imwrite(os.path.join(tmp_dir, f"f_{idx:04d}.png"), frame)

    first_idx = frame_indices[0]
    cmd = (f'ffmpeg -y -framerate {fps} -start_number {first_idx} '
           f'-i "{tmp_dir}/f_%04d.png" -c:v libx264 -pix_fmt yuv420p '
           f'-vf "setpts={slow_factor}*PTS" "{out_path}"')
    ret = os.system(cmd + " > /dev/null 2>&1")

    # cleanup temp frames
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)

    return ret == 0
