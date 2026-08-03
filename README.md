# Automated Ball Tracking & Trajectory Estimator

Ball tracking + trajectory visualization for sports footage, built and
validated against real, messy broadcast video -- not just clean lab clips.

**Live demo:** (Hugging Face Space link goes here once deployed)
**Portfolio:** mohammadzaidshaikh.com

## What this does

1. **Shot detection** -- splits input video into distinct camera shots
   (hard cuts), since highlight/edit footage is almost never one
   continuous take. Tracking never runs across a cut.
2. **Ball detection** -- combines motion differencing and color cues
   (HSV) to find candidate ball blobs per frame.
3. **Kalman-gated tracking** -- a constant-velocity Kalman filter
   predicts the ball's next position and only accepts detector
   candidates that fall within a physically plausible gate, rejecting
   background clutter.
4. **Human-in-the-loop confirmation** -- for footage too cluttered for
   reliable automatic detection (see below), the app lets you confirm
   the ball's position on a handful of checkpoint frames instead of
   silently trusting noisy auto-detections.
5. **Segment-aware trajectory** -- interpolates between confirmed
   points per physically distinct segment (e.g. before/after bat
   contact), rather than one global constant-velocity fit that would
   break across a velocity discontinuity.
6. **Physics summary** -- pixel-space speed and direction per segment,
   with an explicit note about what would be needed to convert to
   real-world units (mph, m/s).

## Why human-in-the-loop, not fully automatic?

This was validated against a real CPL broadcast clip (a diving
boundary catch). Automatic detection struggled for reasons that are
common to broadcast sports footage in general, not specific to one bad
video:

- **Camera pan/zoom** following the action breaks the static-background
  assumption behind classical motion detection.
- **Animated LED advertising boards** introduce motion/color noise
  unrelated to anything in play.
- **A small, fast, motion-blurred ball** against grass and a crowded
  background is a genuinely hard detection target for classical CV
  without a trained model.
- **Bat contact (or any impact) introduces a hard velocity
  discontinuity** -- a single trajectory model across it will diverge.

Rather than paper over these with a detector that looks like it's
working but silently produces wrong tracks, this project makes the
confirmation step an explicit, fast part of the workflow: click the
ball on ~5 frames, and the pipeline handles the interpolation, segment
detection, and rendering. On cleaner footage (static camera, high
contrast ball, no impact discontinuities) the Kalman-gated tracker
works with fewer or no manual confirmations needed.

## Project structure

```
pipeline/
  shot_detect.py      # hard-cut / shot boundary detection
  detector.py          # motion + color candidate blob detection
  tracker.py           # Kalman filter with gated data association
  trajectory.py        # anchor interpolation, physics, video rendering
  app.py               # Gradio interface
  requirements.txt
```

## Running locally

```bash
pip install -r pipeline/requirements.txt
python pipeline/app.py
```

## Deployment

Deployed via Hugging Face Spaces (Gradio SDK). Push this repo's
`pipeline/` contents to a Space; HF handles the rest. See the
portfolio site for a live link and write-up.
