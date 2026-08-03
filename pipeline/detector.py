"""
Per-frame ball candidate detector.

Combines two independent, weak cues that are individually too noisy for
this footage (crowd, animated LED boards, camera pan, light-colored kit)
but together give reasonably high precision:

  1. Motion: consecutive-frame differencing (simple, since motion
     compensation on this footage gave limited additional benefit over
     the cost -- documented during development).
  2. Color: the ball is white under floodlights -> low saturation,
     high value in HSV.

Candidates are the AND of both masks, filtered by size and circularity.
Ambiguity in the resulting candidate list is resolved downstream by the
Kalman-gated tracker, not here -- this module intentionally returns
*all* plausible candidates per frame, not a single answer.
"""
import cv2
import numpy as np
import os


TOP_BANNER_CUTOFF = 210  # static broadcast overlay, excluded from all detection


def get_candidates(frame, prev_frame):
    """Returns list of (x, y, radius, area, circularity) for one frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(motion, 20, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255))

    combined = cv2.bitwise_and(motion_mask, color_mask)
    combined[0:TOP_BANNER_CUTOFF, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 2 or area > 150:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.35:
            continue
        (x, y), radius = cv2.minEnclosingCircle(c)
        candidates.append((x, y, radius, area, circularity))

    return candidates


def detect_all(frames_dir, start_idx, end_idx):
    """Run candidate detection across a frame range. Returns dict frame_idx -> candidates."""
    results = {}
    prev = None
    for idx in range(start_idx, end_idx + 1):
        path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
        frame = cv2.imread(path)
        if frame is None:
            continue
        if prev is not None:
            results[idx] = get_candidates(frame, prev)
        prev = frame
    return results


if __name__ == "__main__":
    res = detect_all("frames_full", 125, 231)
    total = sum(len(v) for v in res.values())
    avg = total / len(res)
    print(f"Frames processed: {len(res)}, total candidates: {total}, avg/frame: {avg:.1f}")
    # show distribution
    counts = [len(v) for v in res.values()]
    print("frames with 0 candidates:", sum(1 for c in counts if c == 0))
    print("frames with 1-3 candidates:", sum(1 for c in counts if 1 <= c <= 3))
    print("frames with >3 candidates:", sum(1 for c in counts if c > 3))
