# app.py
# Streamlit app: text -> frames (Stable Diffusion) -> video (ffmpeg)
# Produces custom, (model-license-dependent) copyright-free videos.
# WARNING: This is computationally heavy and requires a GPU for usable speed.

import os
import tempfile
import time
from pathlib import Path
from io import BytesIO

import streamlit as st
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import ffmpeg

# Diffusers imports
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

st.set_page_config(page_title="Text → Video (Frame-based) Generator", layout="wide")

# ----------------------------
# SETTINGS / Defaults
# ----------------------------
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"  # change as needed
FPS = 6  # frames per second (low to reduce compute)
WIDTH = 512
HEIGHT = 512
FRAMES_LIMIT = 90  # safety upper bound (e.g., 15s @ 6fps = 90 frames)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_sd_pipeline(model_id: str, device: str = "cuda"):
    """
    Load a Stable Diffusion pipeline. Use cached resource to avoid reloading.
    """
    # Use DPMSolverMultistepScheduler for faster denoising steps
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    # reduce memory footprint slightly
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

def generate_frame(pipe, prompt, seed, guidance_scale=7.5, num_inference_steps=20, strength=0.7, image_size=(WIDTH, HEIGHT)):
    """
    Generate one image frame from prompt using stable-diffusion text2img.
    For slightly better continuity, we can vary seed/prompt slightly per frame.
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    with torch.autocast(pipe.device.type):
        out = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=image_size[1],
            width=image_size[0],
            generator=generator,
        )
    pil_img = out.images[0]
    return pil_img

def assemble_video_ffmpeg(frames_dir: Path, out_path: Path, fps: int = FPS, crf=18):
    """
    Use ffmpeg to assemble frames saved like frame_0001.png into an mp4.
    """
    (
        ffmpeg
        .input(str(frames_dir / "frame_%05d.png"), framerate=fps, pattern_type="sequence")
        .output(str(out_path), vcodec="libx264", preset="medium", crf=crf, pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🪄 Text → Video (Frame-by-frame) Generator")
st.caption(
    "Generates short videos by producing frames from a text prompt (Stable Diffusion) and stitching them with ffmpeg. "
    "Requires significant compute (GPU recommended). Check model license before commercial use."
)

col1, col2 = st.columns([3, 1])

with col1:
    prompt = st.text_area("Describe your scene (prompt):", value="A serene neon forest with glowing butterflies at dawn, cinematic")
    duration_sec = st.number_input("Duration (seconds)", min_value=2, max_value=60, value=6, step=1, help="Shorter durations are much faster.")
    fps = st.slider("Frames per second (fps)", min_value=1, max_value=12, value=FPS)
    model_id = st.text_input("Diffusion model (Hugging Face id)", value=DEFAULT_MODEL_ID)
    guidance = st.slider("Guidance scale (higher = more prompt-following)", min_value=3.0, max_value=15.0, value=7.5)
    steps = st.slider("Sampling steps (quality vs speed)", min_value=10, max_value=50, value=20)
    seed = st.number_input("Base random seed (integer, 0 for random)", min_value=0, value=0, step=1)

with col2:
    st.markdown("**Hints & Warnings**")
    st.info("GPU recommended. Generating many frames on CPU = very slow.")
    st.markdown("- Lower fps (e.g., 4-6) → fewer frames → faster\n- Use short durations (2-10s) for quick results\n- Check the chosen model license before commercial use")

generate_btn = st.button("✨ Generate video")

# ----------------------------
# Main flow
# ----------------------------
if generate_btn:
    if duration_sec < 1:
        st.error("Duration must be at least 1 second.")
    else:
        num_frames = int(duration_sec * fps)
        if num_frames > FRAMES_LIMIT:
            st.warning(f"Requested {num_frames} frames exceeds safety limit ({FRAMES_LIMIT}). Clamping.")
            num_frames = FRAMES_LIMIT

        # Device check
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"Running on device: **{device}**")
        if device == "cpu":
            st.warning("CPU detected. This will be very slow and may not finish for many frames. Consider using GPU.")

        # Load pipeline
        try:
            with st.spinner("Loading model pipeline (this may take ~30s the first time)..."):
                pipe = load_sd_pipeline(model_id, device=device)
        except Exception as e:
            st.error(f"Failed to load model '{model_id}'. Error: {e}")
            st.stop()

        # Prepare temporary folder for frames and output
        tmpdir = Path(tempfile.mkdtemp(prefix="txt2vid_"))
        frames_dir = tmpdir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        out_video_path = tmpdir / "output.mp4"

        # If seed == 0 use random base
        base_seed = int(time.time()) if int(seed) == 0 else int(seed)
        st.write(f"Base seed: {base_seed}  |  Frames: {num_frames}  |  Size: {WIDTH}x{HEIGHT}")

        progress_bar = st.progress(0)
        status = st.empty()
        frame_paths = []

        # Generate frames. For simple continuity, we keep same seed but vary it a bit per frame.
        for i in range(num_frames):
            frame_seed = base_seed + i  # simple deterministic variation
            status.text(f"Generating frame {i+1}/{num_frames} (seed={frame_seed})...")
            try:
                pil = generate_frame(pipe, prompt, frame_seed, guidance_scale=guidance, num_inference_steps=steps)
            except Exception as e:
                st.error(f"Frame generation failed at frame {i+1}: {e}")
                break

            frame_name = frames_dir / f"frame_{i:05d}.png"
            pil.save(frame_name)
            frame_paths.append(frame_name)
            progress_bar.progress((i+1) / num_frames)

        # Assemble video if all frames generated
        if len(frame_paths) == num_frames:
            status.text("Assembling video with ffmpeg...")
            try:
                assemble_video_ffmpeg(frames_dir, out_video_path, fps=fps)
                status.success("Video assembled ✔️")
                st.video(str(out_video_path))
                with open(out_video_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button("⬇️ Download video (MP4)", data=video_bytes, file_name="generated_video.mp4", mime="video/mp4")
                st.info(f"Frames and output are in: `{tmpdir}` (auto-created).")
            except Exception as e:
                st.error(f"Failed to assemble video: {e}")
        else:
            st.error("Not all frames were generated — aborting assembly.")

