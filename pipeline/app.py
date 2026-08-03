"""
Ball Tracking & Trajectory Estimator -- Gradio app.

Workflow:
  1. Upload a video clip.
  2. The app splits it into shots (hard cuts) and lets you pick which
     one to track -- essential for highlight/edit footage, which is
     almost never one continuous take.
  3. It steps through a handful of evenly-spaced checkpoint frames from
     that shot, one at a time. Click the ball's position on each (or
     skip if it's not visible -- e.g. occluded by a catch).
  4. It interpolates a physically-grounded trajectory between your
     confirmed points, reports segment-wise speed/direction (revealing
     things like bat-contact direction changes), and renders an
     annotated output video.

Why click-to-confirm rather than fully automatic detection? Broadcast
sports footage is often too cluttered (crowd, animated ad boards,
camera pan, light-colored kit) for classical CV to isolate a small,
fast, motion-blurred ball reliably on every frame. Rather than present
unreliable auto-detections as ground truth, this app makes the human
confirmation step explicit and fast (a handful of clicks), and is
upfront that this is a semi-automatic tool for hard footage -- not a
production multi-camera tracking system.
"""
import gradio as gr
import cv2
import os
import shutil
import tempfile
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shot_detect import detect_shots
from trajectory import build_track, segment_physics, render_video

WORKDIR = tempfile.mkdtemp(prefix="ball_tracker_")
FRAMES_DIR = os.path.join(WORKDIR, "frames")
N_CHECKPOINTS = 5


def extract_frames(video_path):
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR)
    os.system(f'ffmpeg -y -i "{video_path}" "{FRAMES_DIR}/frame_%04d.png" > /dev/null 2>&1')
    return len([f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")])


def load_frame_rgb(frame_idx):
    img = cv2.imread(os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.png"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def on_upload(video_path):
    if video_path is None:
        return gr.update(choices=[], value=None), "Upload a video to begin.", []
    n_frames = extract_frames(video_path)
    shots, _ = detect_shots(FRAMES_DIR)
    choices = [f"Shot {i}: frames {s}-{e} ({e - s + 1} frames)" for i, (s, e) in enumerate(shots)]
    msg = f"Found {len(shots)} shot(s) across {n_frames} frames. Pick one to track, then click Start."
    return gr.update(choices=choices, value=choices[0] if choices else None), msg, shots


def start_checkpointing(shot_label, shots):
    idx = int(shot_label.split(":")[0].replace("Shot ", ""))
    start, end = shots[idx]
    checkpoints = [start + int(i * (end - start) / (N_CHECKPOINTS - 1)) for i in range(N_CHECKPOINTS)]
    first_img = load_frame_rgb(checkpoints[0])
    msg = f"Frame {checkpoints[0]} (checkpoint 1 of {N_CHECKPOINTS}). Click the ball, or click Skip if not visible."
    return checkpoints, [], 0, first_img, msg, gr.update(visible=False)


def record_click(evt: gr.SelectData, checkpoints, clicks, current_idx):
    x, y = evt.index[0], evt.index[1]
    clicks = clicks + [(checkpoints[current_idx], x, y)]
    return advance(checkpoints, clicks, current_idx)


def record_skip(checkpoints, clicks, current_idx):
    return advance(checkpoints, clicks, current_idx)


def advance(checkpoints, clicks, current_idx):
    current_idx += 1
    if current_idx >= len(checkpoints):
        msg = f"All {len(checkpoints)} checkpoints reviewed ({len(clicks)} confirmed). Click Build Trajectory."
        return checkpoints, clicks, current_idx, None, msg, gr.update(visible=True)
    img = load_frame_rgb(checkpoints[current_idx])
    msg = f"Frame {checkpoints[current_idx]} (checkpoint {current_idx + 1} of {len(checkpoints)})."
    return checkpoints, clicks, current_idx, img, msg, gr.update(visible=False)


def build_and_render(clicks):
    if len(clicks) < 2:
        return None, "Need at least 2 confirmed ball positions to build a trajectory."

    last_frame = clicks[-1][0]
    track = build_track(clicks, extrapolate_to=last_frame)
    physics = segment_physics(clicks)

    out_path = os.path.join(WORKDIR, "tracked_output.mp4")
    ok = render_video(FRAMES_DIR, track, out_path)
    if not ok:
        return None, "Rendering failed -- check ffmpeg is available."

    lines = ["Segment-wise speed & direction (pixel-space):"]
    for seg in physics:
        lines.append(f"  frames {seg['from_frame']}-{seg['to_frame']}: "
                      f"{seg['speed_px_per_frame']} px/frame at {seg['direction_deg']}\u00b0")
    lines.append("")
    lines.append("Note: pixel/frame speeds, not mph -- real-world units need camera "
                  "calibration and the source clip's slow-motion factor, which can't be "
                  "inferred from a single uncalibrated video.")
    return out_path, "\n".join(lines)


with gr.Blocks(title="Ball Tracking & Trajectory Estimator") as demo:
    gr.Markdown("# Automated Ball Tracking & Trajectory Estimator")
    gr.Markdown(
        "Upload a clip, pick the shot to analyze, confirm the ball's position on a few "
        "checkpoint frames, and get back an annotated trajectory with segment speed/direction. "
        "Built to handle real, messy broadcast footage (camera pans, crowd clutter, animated "
        "ad boards, bat-contact direction changes) rather than only clean lab video."
    )

    shots_state = gr.State([])
    checkpoints_state = gr.State([])
    clicks_state = gr.State([])
    current_idx_state = gr.State(0)

    video_in = gr.Video(label="Upload clip")
    status_md = gr.Markdown("Upload a video to begin.")
    shot_dropdown = gr.Dropdown(label="Select shot to track", choices=[])
    start_btn = gr.Button("Start checkpointing this shot")

    checkpoint_image = gr.Image(label="Click the ball's position", interactive=False)
    skip_btn = gr.Button("Skip (ball not visible in this frame)")

    build_btn = gr.Button("Build Trajectory", variant="primary", visible=False)
    output_video = gr.Video(label="Annotated trajectory")
    physics_report = gr.Textbox(label="Physics summary", lines=8)

    video_in.change(on_upload, inputs=video_in, outputs=[shot_dropdown, status_md, shots_state])
    start_btn.click(start_checkpointing, inputs=[shot_dropdown, shots_state],
                     outputs=[checkpoints_state, clicks_state, current_idx_state,
                              checkpoint_image, status_md, build_btn])
    checkpoint_image.select(record_click, inputs=[checkpoints_state, clicks_state, current_idx_state],
                             outputs=[checkpoints_state, clicks_state, current_idx_state,
                                      checkpoint_image, status_md, build_btn])
    skip_btn.click(record_skip, inputs=[checkpoints_state, clicks_state, current_idx_state],
                    outputs=[checkpoints_state, clicks_state, current_idx_state,
                             checkpoint_image, status_md, build_btn])
    build_btn.click(build_and_render, inputs=[clicks_state], outputs=[output_video, physics_report])


if __name__ == "__main__":
    demo.launch()
