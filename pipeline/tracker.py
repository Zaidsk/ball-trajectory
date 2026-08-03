"""
Kalman-gated tracker.

The detector returns many plausible blobs per frame (noisy broadcast
footage: crowd, LED boards, light-colored kit). This module resolves
that ambiguity using a constant-velocity Kalman filter: it predicts
where the ball should be next, then accepts only the candidate that
falls within a gating radius of that prediction. Frames with no
candidate inside the gate are treated as a miss (occlusion / detector
failure) and the filter coasts on its prediction, with the gate
widening the longer it goes without a real measurement.
"""
import numpy as np
import cv2


class BallTracker:
    def __init__(self, seed_x, seed_y, seed_vx=0.0, seed_vy=0.0,
                 base_gate_radius=30, max_gate_radius=70, process_noise=6.0,
                 measurement_noise=6.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.statePost = np.array([[seed_x], [seed_y], [seed_vx], [seed_vy]], dtype=np.float32)

        self.base_gate_radius = base_gate_radius
        self.max_gate_radius = max_gate_radius
        self.misses = 0

    def current_gate_radius(self):
        return min(self.base_gate_radius + self.misses * 15, self.max_gate_radius)

    def step(self, candidates):
        """
        candidates: list of (x, y, radius, area, circularity) for this frame.
        Returns (x, y, matched: bool) for this frame's tracked position.
        """
        pred = self.kf.predict()
        pred_x, pred_y = float(pred[0].item()), float(pred[1].item())

        gate = self.current_gate_radius()
        best = None
        best_dist = gate
        for (x, y, r, area, circ) in candidates:
            dist = np.hypot(x - pred_x, y - pred_y)
            if dist <= best_dist:
                best = (x, y)
                best_dist = dist

        if best is not None:
            measurement = np.array([[np.float32(best[0])], [np.float32(best[1])]])
            self.kf.correct(measurement)
            self.misses = 0
            corrected = self.kf.statePost
            return float(corrected[0].item()), float(corrected[1].item()), True
        else:
            self.misses += 1
            return pred_x, pred_y, False


def run_tracker(frames_dir, candidates_by_frame, start_idx, end_idx,
                 seed_x, seed_y, seed_vx=0.0, seed_vy=0.0):
    tracker = BallTracker(seed_x, seed_y, seed_vx, seed_vy)
    track = []
    for idx in range(start_idx, end_idx + 1):
        cands = candidates_by_frame.get(idx, [])
        x, y, matched = tracker.step(cands)
        track.append({"frame": idx, "x": x, "y": y, "matched": matched,
                       "gate": tracker.current_gate_radius()})
    return track
