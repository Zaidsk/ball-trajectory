"""
Shot boundary detection.

Broadcast sports footage — especially highlight/edit clips — is almost
never a single continuous camera take. Background subtraction and
frame-to-frame tracking both assume temporal continuity, so running
them across a hard cut produces garbage. This module segments a frame
sequence into shots using HSV color-histogram correlation between
consecutive frames, so the rest of the pipeline can process each shot
independently and pick (or let the user pick) the one to track.
"""
import cv2
import numpy as np
import os


def detect_shots(frames_dir, corr_threshold=0.7, bins=16):
    """
    Returns a list of (start_idx, end_idx) inclusive frame-index tuples,
    one per detected shot, plus the raw per-frame correlation scores.
    """
    frame_files = sorted(os.listdir(frames_dir))
    n = len(frame_files)

    correlations = [None]  # correlations[i] = corr(frame i-1, frame i)
    prev_hist = None
    cut_points = []

    for i, fname in enumerate(frame_files):
        img = cv2.imread(os.path.join(frames_dir, fname))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins],
                             [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            correlations.append(corr)
            if corr < corr_threshold:
                cut_points.append(i)
        prev_hist = hist

    shots = []
    start = 0
    for cut in cut_points:
        shots.append((start, cut - 1))
        start = cut
    shots.append((start, n - 1))

    return shots, correlations


if __name__ == "__main__":
    shots, corrs = detect_shots("frames_full")
    print(f"Detected {len(shots)} shots:")
    for i, (s, e) in enumerate(shots):
        print(f"  Shot {i}: frames {s}-{e}  ({e - s + 1} frames, {(e - s + 1) / 25:.2f}s @ 25fps)")
