"""Graphical User Interface for the FluxPipeline Image Generation System.

This module provides a comprehensive web-based interface for the FluxPipeline
system using Gradio. Features include:
- Single image and GIF generation
- Batch processing capabilities
- Advanced parameter controls
- Memory optimization settings
- Image enhancement options
- Generation history management
- Example prompt library
- Real-time progress tracking

The interface is designed to be user-friendly while providing access to
advanced features for experienced users.

Example:
    Run the GUI application:
    ```bash
    python gui.py --port 7860 --share
    ```

Note:
    The interface automatically handles resource management and provides
    real-time feedback during generation.
"""

import os
import json
import threading
from queue import Queue
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import gradio as gr
from PIL import Image, ImageEnhance
import torch

from config.env_config import setup_environment
from pipeline.flux_pipeline import FluxPipeline
from core.seed_manager import SeedProfile
from utils.system_utils import setup_workspace, suppress_warnings
from config.logging_config import logger

# Initialize environment and pipeline
suppress_warnings()
setup_environment()
workspace = setup_workspace()
pipeline = FluxPipeline(workspace=workspace)

# Load model at startup
if not pipeline.load_model():
    logger.error("Failed to load the model. Exiting.")
    exit(1)

# Initialize history storage
HISTORY_FILE = workspace / "history.json"


def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []


def save_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# Example prompts with categories
EXAMPLE_PROMPTS = {
    "Landscapes": [
        "High-resolution photograph of a serene mountain lake at sunset, snow-capped peaks reflecting perfectly in the crystal-clear water, dramatic sky with vibrant colors, ultra-wide angle, 8K resolution, masterpiece.",
        "Dense rainforest with rays of sunlight filtering through the lush green canopy, mist rising from the ground, exotic birds flying, highly detailed, realistic, professional photography.",
        "Rolling hills covered in lavender fields under a cloudy sky, soft lighting, gentle breeze creating waves in the lavender, impressionist painting style, high detail, vibrant colors.",
    ],
    "Sci-Fi": [
        "Futuristic cityscape with towering skyscrapers, flying vehicles weaving between buildings, neon advertisements illuminating the streets below, cyberpunk aesthetic, ultra-realistic, cinematic lighting.",
        "An abandoned space station orbiting a distant purple planet, stars and galaxies visible in the background, haunting atmosphere, highly detailed, 4K resolution.",
        "Cyberpunk street market bustling with activity, holographic vendors displaying wares, robot customers interacting, rain-soaked streets reflecting neon lights, dynamic composition, high detail.",
    ],
    "Fantasy": [
        "Ancient dragon perched atop a crystal castle tower, wings spread wide, scales glistening in the moonlight, dramatic clouds in the night sky, epic fantasy art, highly detailed.",
        "Mystical forest clearing with glowing mushrooms and fairy lights, a small fairy sitting on a mushroom, ethereal atmosphere, soft lighting, high detail, fantasy illustration.",
        "Wizard's study filled with floating books and magical artifacts, candles casting warm light, intricate details on scrolls and potions, high-resolution digital art, rich colors.",
    ],
    "Portraits": [
        "High-quality portrait of a beautiful gothic woman with a voluptuous figure, pale skin, dark makeup, and long black hair. She is wearing elegant gothic attire with intricate lace and accessories. The setting has a moody, atmospheric background with dramatic lighting to enhance the overall aesthetic. Realistic, highly detailed, professional photography, 4K resolution, masterpiece.",
        "Portrait of an elderly man with deep wrinkles, wearing a weathered hat, eyes full of wisdom, background blurred softly to focus on the subject, highly detailed, realistic, professional lighting.",
        "Close-up portrait of a futuristic cyborg with glowing eyes, metallic skin with intricate circuitry patterns, dark background with sparks flying, ultra-realistic, high detail, 8K resolution.",
    ],
    "Abstract": [
        "Swirling vibrant colors representing the concept of time and space, fractal patterns, cosmic energy, high-resolution digital art, dynamic composition.",
        "Geometric patterns morphing into organic shapes, blending of sharp lines and soft curves, monochromatic color scheme, minimalist design, high resolution.",
        "Abstract representation of human emotions using bold colors and textures, expressive brush strokes, modern art style, large canvas, gallery quality.",
    ],
    "GIF Sequences": [
        "A seed sprouting from the soil || A small plant emerging || The plant growing leaves || A blooming flower || The flower wilting",
        "Sun rising over the mountains || Midday sun at its peak || Sun setting behind the hills || Night sky full of stars || Dawn breaking",
        "Sketch of a car || Adding color to the car || Car transforming into a futuristic vehicle || Vehicle lifting off as a spaceship",
        "Empty canvas || Brush strokes appearing || Landscape taking shape || Colors filling in || Final masterpiece revealed",
        "Caterpillar on a leaf || Forming a chrysalis || Emerging butterfly || Butterfly flying away || Flowers blooming in its path",
    ],
}

# Custom CSS styles
CUSTOM_CSS = """
/* Custom CSS styles for enhanced UI appearance */
:root {
    --primary-color: #3498db;
    --secondary-color: #2ecc71;
    --background-color: #f0f2f5;
    --text-color: #333333;
    --border-color: #e0e0e0;
}

body {
    font-family: 'Helvetica', sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.header {
    text-align: center;
    margin-bottom: 2rem;
}

.header h1 {
    font-size: 2.5rem;
    color: var(--primary-color);
}

.header p {
    font-size: 1.2rem;
    color: var(--text-color);
    opacity: 0.8;
}

.card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.input-group {
    margin-bottom: 1rem;
}

.input-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

.input-group input[type="text"],
.input-group textarea {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 1rem;
}

.input-group input[type="range"] {
    width: 100%;
}

.button-primary {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.button-primary:hover {
    background-color: #2980b9;
}

.output-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 300px;
    background-color: #f8f9fa;
    border-radius: 8px;
    overflow: hidden;
}

.output-container img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

.loading-animation {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 200px;
}

.loading-spinner {
    width: 50px;
    height: 50px;
    border: 5px solid #f3f3f3;
    border-top: 5px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

.loading-text {
    margin-top: 20px;
    font-size: 18px;
    color: var(--primary-color);
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

.gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
}

.gallery img {
    width: 100%;
    height: 200px;
    object-fit: cover;
    border-radius: 4px;
}

.alert {
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 1rem;
}

.alert-error {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
}

.alert-success {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}

/* Additional styles for enhanced features */
.batch-container {
    background: rgba(0,0,0,0.05);
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.memory-settings {
    border-left: 3px solid var(--primary-color);
    padding-left: 1rem;
    margin: 1rem 0;
}

.progress-bar {
    height: 4px;
    background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
    width: 0%;
    transition: width 0.3s ease;
}

.history-card {
    display: flex;
    gap: 1rem;
    padding: 1rem;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    margin-bottom: 1rem;
}

.history-card img {
    max-width: 200px;
    border-radius: 4px;
}

.history-info {
    flex: 1;
}

.error-message {
    background-color: #fee;
    border-left: 4px solid #f44;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
}
"""


def get_loading_animation() -> str:
    """Create HTML loading animation.

    Returns:
        str: HTML string containing loading animation
    """
    return """
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">Generating...</div>
        </div>
    """


def create_optimized_gif(
    images: List[Image.Image], output_path: Path, duration: int = 500
):
    """Create an optimized GIF from a list of images.

    Args:
        images (List[Image.Image]): List of PIL images
        output_path (Path): Path to save the GIF
        duration (int, optional): Frame duration in ms. Defaults to 500.
    """
    if not images:
        return None

    # Optimize images for GIF format
    optimized_images = []
    for img in images:
        opt_img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
        optimized_images.append(opt_img)

    # Save optimized GIF
    optimized_images[0].save(
        output_path,
        save_all=True,
        append_images=optimized_images[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )


def apply_upscaling(image: Image.Image, upscale_factor: int = 2) -> Image.Image:
    """Apply upscaling to improve image quality.

    Args:
        image (Image.Image): Input image
        upscale_factor (int, optional): Upscaling factor. Defaults to 2.

    Returns:
        Image.Image: Upscaled image
    """
    new_size = (image.width * upscale_factor, image.height * upscale_factor)
    upscaled_image = image.resize(new_size, resample=Image.LANCZOS)
    logger.info(f"Image upscaled to {new_size}")
    return upscaled_image


def enhance_faces(image: Image.Image) -> Image.Image:
    """Enhance facial features in the image.

    Args:
        image (Image.Image): Input image

    Returns:
        Image.Image: Enhanced image
    """
    # Apply sharpness enhancement
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)

    # Apply contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    logger.info("Face enhancement applied")
    return image


def export_image(
    image: Image.Image, format: str, quality: int, seed: int, index: int = 0
) -> Path:
    """Export image with specified format and quality.

    Args:
        image (Image.Image): Image to export
        format (str): Export format (PNG, JPEG, WEBP, GIF)
        quality (int): Export quality (1-100)
        seed (int): Generation seed for filename
        index (int, optional): Image index. Defaults to 0.

    Returns:
        Path: Path to exported image

    Raises:
        ValueError: If format is not supported
    """
    # Create export directory
    output_dir = workspace / "exports"
    output_dir.mkdir(exist_ok=True)

    # Generate output path
    output_path = (
        output_dir / f"image_generated_{index:03}_using_seed-{seed}.{format.lower()}"
    )

    # Export based on format
    if format.upper() == "PNG":
        image.save(output_path, "PNG", optimize=True)
    elif format.upper() in ["JPEG", "JPG"]:
        image.save(output_path, "JPEG", quality=quality, optimize=True)
    elif format.upper() == "WEBP":
        image.save(output_path, "WEBP", quality=quality, method=6)
    elif format.upper() == "GIF":
        image.save(output_path, "GIF")
    else:
        raise ValueError(f"Unsupported export format: {format}")

    logger.info(f"Image exported to {output_path}")
    return output_path


def generate_image(
    prompt: str,
    negative_prompt: str,
    seed_profile: str,
    manual_seed: str,
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    generation_mode: str,
    generate_all_profiles: bool,
    batch_size: int,
    batch_variation: str,
    enable_attention_slicing: bool,
    enable_cpu_offload: bool,
    use_upscaling: bool = False,
    enable_face_enhancement: bool = False,
    export_format: str = "PNG",
    export_quality: int = 95,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate images based on user inputs.

    This function handles the complete image generation process including:
    - Input validation and processing
    - Single image and GIF generation
    - Batch processing
    - Image enhancement
    - Export handling
    - Progress tracking

    Args:
        prompt (str): Main generation prompt
        negative_prompt (str): Elements to avoid
        seed_profile (str): Selected seed profile
        manual_seed (str): Optional manual seed
        guidance_scale (float): Generation guidance scale
        num_inference_steps (int): Number of generation steps
        height (int): Output image height
        width (int): Output image width
        generation_mode (str): Single Image or GIF Mode
        generate_all_profiles (bool): Use all seed profiles
        batch_size (int): Number of images to generate
        batch_variation (str): Type of batch variation
        enable_attention_slicing (bool): Memory optimization
        enable_cpu_offload (bool): CPU offload setting
        use_upscaling (bool, optional): Apply upscaling. Defaults to False.
        enable_face_enhancement (bool, optional): Enhance faces. Defaults to False.
        export_format (str, optional): Export format. Defaults to "PNG".
        export_quality (int, optional): Export quality. Defaults to 95.
        progress (gr.Progress, optional): Progress tracker.

    Yields:
        List[gr.update]: Updates for Gradio components
    """
    messages = []  # List to accumulate status messages
    try:
        # Initialize progress
        progress(0, desc="Initializing...")

        # Show loading state
        yield [
            gr.update(visible=False),  # 1. error_box
            gr.update(visible=False),  # 2. output_image
            gr.update(visible=False),  # 3. output_gallery
            gr.update(visible=False),  # 4. gif_output
            gr.update(
                value=get_loading_animation(), visible=True
            ),  # 5. loading_indicator
            gr.update(value="Initializing...", visible=True),  # 6. status_text
            gr.update(value=load_history_gallery()),  # 7. history_gallery
            gr.update(visible=False),  # 8. download_gif_button
            gr.update(visible=False),  # 9. download_image_button
        ]

        # Map seed profiles
        profile_map = {
            "Conservative": SeedProfile.CONSERVATIVE,
            "Balanced": SeedProfile.BALANCED,
            "Experimental": SeedProfile.EXPERIMENTAL,
            "Full Range": SeedProfile.FULL_RANGE,
        }

        # Process manual seed
        seed = None
        if manual_seed.strip():
            try:
                seed = abs(int(manual_seed))
            except ValueError:
                raise ValueError("Invalid seed value. Please enter a positive integer.")

        images = []  # List to store generated images
        messages = []  # List to store status messages

        # Configure pipeline options
        pipeline.enable_attention_slicing = enable_attention_slicing
        pipeline.enable_cpu_offload = enable_cpu_offload

        if generation_mode == "GIF Mode":
            # GIF Mode: Generate an animated GIF from multiple prompts

            # Split the prompt into individual prompts for each frame
            prompts = [p.strip() for p in prompt.split("||") if p.strip()]
            if not prompts:
                raise ValueError("No prompts provided for GIF generation.")

            # If only one prompt is provided, generate multiple frames by varying the seed
            if len(prompts) == 1:
                num_frames = 10  # Default number of frames if only one prompt is given
                prompts = prompts * num_frames  # Repeat the single prompt

            # Use the selected seed profile (do not loop over seed profiles in GIF Mode)
            selected_profile = profile_map.get(seed_profile, SeedProfile.BALANCED)

            total_steps = len(prompts)
            current_step = 0

            for idx, p in enumerate(prompts):
                # Update progress bar
                progress(
                    current_step / total_steps * 100,
                    desc=f"Generating frame {idx+1}/{len(prompts)}...",
                )

                # Vary the seed per frame
                frame_seed = None if seed is None else seed + idx

                # Generate image for the current frame
                image, used_seed = pipeline.generate_image(
                    prompt=p,
                    negative_prompt=negative_prompt,
                    seed_profile=selected_profile,
                    seed=frame_seed,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                )
                current_step += 1

                if image:
                    # Apply enhancements if enabled
                    if use_upscaling:
                        image = apply_upscaling(image)
                    if enable_face_enhancement:
                        image = enhance_faces(image)
                    # Add image to the list of frames
                    images.append(image)
                    messages.append(f"Frame {idx+1} generated with seed {used_seed}.")
                else:
                    messages.append(f"Failed to generate frame {idx+1}.")

            if images:
                # Generate a unique filename for the GIF
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_gif_path = Path(workspace) / f"generated_gif_{timestamp}.gif"

                # Create the optimized GIF from the frames
                create_optimized_gif(images, output_gif_path)

                gif_image_path = str(output_gif_path)

                # Save to history
                save_history(
                    {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "generation_mode": generation_mode,
                        "image_path": gif_image_path,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Yield the updated UI components
                yield [
                    gr.update(visible=False),  # error_box
                    gr.update(visible=False),  # output_image
                    gr.update(visible=False),  # output_gallery
                    gr.update(
                        value=gif_image_path,
                        visible=True,
                    ),  # gif_output
                    gr.update(visible=False),  # loading_indicator
                    gr.update(value="\n".join(messages), visible=True),  # status_text
                    gr.update(value=load_history_gallery()),  # history_gallery
                    gr.update(
                        value=gif_image_path,
                        visible=True,
                    ),  # download_gif_button
                    gr.update(visible=False),  # download_image_button
                ]
            else:
                # Handle case where no images were generated
                messages.append("No frames were generated for the GIF.")
                yield [
                    gr.update(visible=False),  # 1. error_box
                    gr.update(visible=False),  # 2. output_image
                    gr.update(visible=False),  # 3. output_gallery
                    gr.update(visible=False),  # 4. gif_output
                    gr.update(visible=False),  # 5. loading_indicator
                    gr.update(
                        value="\n".join(messages), visible=True
                    ),  # 6. status_text
                    gr.update(value=load_history_gallery()),  # 7. history_gallery
                    gr.update(visible=False),  # 8. download_gif_button
                    gr.update(visible=False),  # 9. download_image_button
                ]

        else:
            # Single Image Mode: Generate one or multiple images

            # Determine the seed profiles to use
            seed_profiles = (
                list(profile_map.values())
                if generate_all_profiles
                else [profile_map.get(seed_profile, SeedProfile.BALANCED)]
            )

            total_steps = batch_size * len(seed_profiles)
            current_step = 0

            for b in range(batch_size):
                # Determine the prompt for the current batch item
                batch_prompt = prompt
                if batch_variation == "Prompt Variations":
                    batch_prompt = f"{prompt} variation {b+1}"

                for profile in seed_profiles:
                    # Update progress bar
                    progress(
                        current_step / total_steps * 100,
                        desc=f"Generating image {current_step+1}/{total_steps}...",
                    )

                    # Vary the seed per image in the batch if manual seed is not provided
                    image_seed = seed
                    if seed is None:
                        image_seed = None  # Random seed
                    else:
                        image_seed = seed + current_step  # Vary seed per image

                    # Generate the image
                    image, used_seed = pipeline.generate_image(
                        prompt=batch_prompt,
                        negative_prompt=negative_prompt,
                        seed_profile=profile,
                        seed=image_seed,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        height=height,
                        width=width,
                    )
                    current_step += 1

                    if image:
                        # Apply enhancements if enabled
                        if use_upscaling:
                            image = apply_upscaling(image)
                        if enable_face_enhancement:
                            image = enhance_faces(image)
                        images.append(
                            (image, f"Seed: {used_seed}, Profile: {profile.name}")
                        )
                        messages.append(
                            f"Image generated with seed {used_seed} using profile {profile.name}."
                        )
                        # Save to history
                        export_path = export_image(
                            image,
                            export_format,
                            export_quality,
                            seed=used_seed,
                            index=current_step,
                        )
                        messages.append(f"Image exported to: {export_path}")
                        save_history(
                            {
                                "prompt": batch_prompt,
                                "negative_prompt": negative_prompt,
                                "generation_mode": generation_mode,
                                "image_path": str(export_path),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                    else:
                        messages.append(
                            f"Failed to generate image with profile {profile.name}."
                        )

            # Handle the outputs
            if images:
                if len(images) > 1:
                    # Multiple images, display in gallery
                    yield [
                        gr.update(visible=False),  # 1. error_box
                        gr.update(visible=False),  # 2. output_image
                        gr.update(
                            value=[img for img, _ in images], visible=True
                        ),  # 3. output_gallery
                        gr.update(visible=False),  # 4. gif_output
                        gr.update(visible=False),  # 5. loading_indicator
                        gr.update(
                            value="\n".join(messages), visible=True
                        ),  # 6. status_text
                        gr.update(value=load_history_gallery()),  # 7. history_gallery
                        gr.update(visible=False),  # 8. download_gif_button
                        gr.update(visible=False),  # 9. download_image_button
                    ]
                else:
                    # Single image
                    image, info = images[0]
                    used_seed = info.split(",")[0].split(":")[1].strip()
                    yield [
                        gr.update(visible=False),  # 1. error_box
                        gr.update(
                            value=image,
                            visible=True,
                        ),  # 2. output_image
                        gr.update(visible=False),  # 3. output_gallery
                        gr.update(visible=False),  # 4. gif_output
                        gr.update(visible=False),  # 5. loading_indicator
                        gr.update(
                            value="\n".join(messages), visible=True
                        ),  # 6. status_text
                        gr.update(value=load_history_gallery()),  # 7. history_gallery
                        gr.update(visible=False),  # 8. download_gif_button
                        gr.update(
                            value=str(export_path),
                            visible=True,
                        ),  # 9. download_image_button
                    ]
            else:
                # Handle case where no images were generated
                messages.append(
                    "No images were generated. Please check your inputs and try again."
                )
                yield [
                    gr.update(visible=False),  # 1. error_box
                    gr.update(visible=False),  # 2. output_image
                    gr.update(visible=False),  # 3. output_gallery
                    gr.update(visible=False),  # 4. gif_output
                    gr.update(visible=False),  # 5. loading_indicator
                    gr.update(
                        value="\n".join(messages), visible=True
                    ),  # 6. status_text
                    gr.update(value=load_history_gallery()),  # 7. history_gallery
                    gr.update(visible=False),  # 8. download_gif_button
                    gr.update(visible=False),  # 9. download_image_button
                ]

            # Update progress bar to 100%
            progress(100, desc="Complete!")

    except Exception as e:
        # Handle errors
        error_msg = f"Error during generation: {str(e)}"
        logger.error(error_msg)
        yield [
            gr.update(
                visible=True, value=f'<div class="error-message">{error_msg}</div>'
            ),  # 1. error_box
            gr.update(visible=False),  # 2. output_image
            gr.update(visible=False),  # 3. output_gallery
            gr.update(visible=False),  # 4. gif_output
            gr.update(visible=False),  # 5. loading_indicator
            gr.update(visible=False),  # 6. status_text
            gr.update(),  # 7. history_gallery
            gr.update(visible=False),  # 8. download_gif_button
            gr.update(visible=False),  # 9. download_image_button
        ]


def load_history_gallery():
    """Load image history for gallery display.

    Returns:
        List[str]: List of image paths from history
    """
    history = load_history()
    return [
        entry["image_path"] for entry in history if Path(entry["image_path"]).exists()
    ]


def create_interface():
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div class="header">
                <h1>🎨 Flux Image Generator</h1>
                <p>Create stunning images and animations using advanced AI models</p>
            </div>
            """
        )

        with gr.Tabs() as tabs:
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column(scale=3):
                        # Main prompt input
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe your image...",
                            lines=4,
                            info="Describe the image you want to generate in detail. For GIF mode, separate multiple prompts with ||.",
                        )

                        # Negative prompt input
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Describe what you don't want in the image...",
                            lines=2,
                            info="Specify elements you want to avoid in the generated image.",
                        )

                        # Enhanced example prompts with categories
                        with gr.Row():
                            prompt_category = gr.Dropdown(
                                choices=list(EXAMPLE_PROMPTS.keys()),
                                label="Prompt Category",
                                info="Select a category of example prompts",
                            )
                            example_prompt = gr.Dropdown(
                                choices=[],
                                label="Example Prompts",
                                info="Select an example prompt or use your own",
                            )

                        def update_example_prompts(category):
                            return gr.update(choices=EXAMPLE_PROMPTS.get(category, []))

                        prompt_category.change(
                            update_example_prompts,
                            inputs=[prompt_category],
                            outputs=[example_prompt],
                        )

                        example_prompt.change(
                            lambda x: x,
                            inputs=[example_prompt],
                            outputs=[prompt],
                        )

                        # Generation modes and seed profiles
                        with gr.Row():
                            generation_mode = gr.Radio(
                                choices=["Single Image", "GIF Mode"],
                                label="Generation Mode",
                                value="Single Image",
                                info="Choose between generating a single image or a GIF sequence.",
                            )
                            seed_profile = gr.Radio(
                                choices=[
                                    "Conservative",
                                    "Balanced",
                                    "Experimental",
                                    "Full Range",
                                ],
                                label="Seed Profile",
                                value="Balanced",
                                info="Select a seed profile to control the randomness of the generation.",
                            )

                        generate_all_profiles = gr.Checkbox(
                            label="Generate for all profiles",
                            value=False,
                            info="Generate images using all seed profiles.",
                        )

                        # Batch Processing
                        with gr.Accordion("Batch Processing", open=False):
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=1,
                                step=1,
                                label="Batch Size",
                                info="Number of images to generate in one batch",
                            )
                            batch_variation = gr.Radio(
                                choices=["Same Prompt", "Prompt Variations"],
                                label="Batch Type",
                                value="Same Prompt",
                                info="Generate multiple images with same prompt or variations",
                            )

                        # Advanced Settings with Memory Optimization
                        with gr.Accordion("Advanced Settings", open=False):
                            manual_seed = gr.Textbox(
                                label="Manual Seed",
                                placeholder="Optional: Enter seed",
                                info="Enter a specific seed for reproducible results.",
                            )

                            with gr.Row():
                                guidance_scale = gr.Slider(
                                    minimum=0,
                                    maximum=20,
                                    value=0.0,
                                    step=0.1,
                                    label="Guidance Scale",
                                    info="Controls how closely the image matches the prompt.",
                                )
                                steps = gr.Slider(
                                    minimum=4,
                                    maximum=50,
                                    value=4,
                                    step=1,
                                    label="Steps",
                                    info="Number of denoising steps. Higher values may produce better quality but take longer.",
                                )

                            with gr.Row():
                                height = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=1024,
                                    step=64,
                                    label="Height",
                                    info="Height of the generated image.",
                                )
                                width = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=1024,
                                    step=64,
                                    label="Width",
                                    info="Width of the generated image.",
                                )

                            with gr.Group(elem_classes="memory-settings"):
                                gr.Markdown("### Memory Optimization")
                                enable_attention_slicing = gr.Checkbox(
                                    label="Enable Attention Slicing",
                                    value=True,
                                    info="Reduces memory usage but may slightly impact generation speed",
                                )
                                enable_cpu_offload = gr.Checkbox(
                                    label="Enable CPU Offload",
                                    value=False,
                                    info="Offload model to CPU when needed (helps with OOM errors)",
                                )

                        with gr.Group():
                            gr.Markdown("### Enhancements")
                            with gr.Row():
                                use_upscaling = gr.Checkbox(
                                    label="Enable Upscaling",
                                    value=False,
                                    info="Apply upscaling to improve image resolution.",
                                )
                                enable_face_enhancement = gr.Checkbox(
                                    label="Enhance Faces",
                                    value=False,
                                    info="Apply face enhancement techniques to improve facial details.",
                                )

                        with gr.Group():
                            gr.Markdown("### Export Options")
                            with gr.Row():
                                export_format = gr.Radio(
                                    choices=["PNG", "JPEG", "WEBP", "GIF"],
                                    label="Format",
                                    value="PNG",
                                    info="Select the export format for your generated image.",
                                )
                                export_quality = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    value=95,
                                    step=1,
                                    label="Quality",
                                    info="Set the export quality (applicable for JPEG and WEBP formats).",
                                )

                        # Progress Indicator
                        progress_bar = gr.HTML(
                            '<div class="progress-bar"></div>',
                            visible=False,
                        )
                        status_text = gr.Textbox(
                            label="Status",
                            interactive=False,
                            placeholder="Generation status messages will appear here.",
                        )

                        # Generate button with enhanced feedback
                        generate_button = gr.Button("Generate", variant="primary")

                        # Error display
                        error_box = gr.HTML(visible=False)

                    with gr.Column(scale=2):
                        download_image_button = gr.DownloadButton(
                            label="Download Image",
                            visible=False,
                        )
                        download_gif_button = gr.DownloadButton(
                            label="Download",
                            visible=False,
                        )
                        output_image = gr.Image(
                            label="Generated Image", type="pil", visible=False
                        )
                        output_gallery = gr.Gallery(
                            label="Generated Images", visible=False
                        )
                        gif_output = gr.Video(
                            label="Generated GIF",
                            visible=False,
                        )
                        loading_indicator = gr.HTML(visible=False)

            # Enhanced History Tab
            with gr.TabItem("History"):
                with gr.Row():
                    history_gallery = gr.Gallery(
                        label="Generation History",
                        show_label=True,
                        elem_id="history_gallery",
                        value=load_history_gallery(),
                    )
                    history_info = gr.JSON(
                        label="Generation Details",
                        visible=True,
                    )

                with gr.Row():
                    clear_history_button = gr.Button("Clear History")
                    export_history_button = gr.Button("Export History")
                    export_status = gr.Markdown()

                def show_history_details(selected_index):
                    history = load_history()
                    if isinstance(selected_index, int) and 0 <= selected_index < len(
                        history
                    ):
                        return history[selected_index]
                    return {}

                history_gallery.select(
                    show_history_details,
                    inputs=[],
                    outputs=[history_info],
                )

                def clear_history():
                    if HISTORY_FILE.exists():
                        HISTORY_FILE.unlink()
                    return (
                        gr.update(value=[]),  # history_gallery
                        gr.update(value={}),  # history_info
                    )

                clear_history_button.click(
                    clear_history,
                    outputs=[history_gallery, history_info],
                )

                def export_history():
                    export_path = workspace / "history_export.json"
                    history = load_history()
                    with open(export_path, "w") as f:
                        json.dump(history, f, indent=2)
                    return f"History exported to {export_path}"

                export_history_button.click(
                    export_history,
                    outputs=[export_status],
                )

            # Event handlers
            generate_button.click(
                fn=generate_image,
                inputs=[
                    prompt,
                    negative_prompt,
                    seed_profile,
                    manual_seed,
                    guidance_scale,
                    steps,
                    height,
                    width,
                    generation_mode,
                    generate_all_profiles,
                    batch_size,
                    batch_variation,
                    enable_attention_slicing,
                    enable_cpu_offload,
                    use_upscaling,
                    enable_face_enhancement,
                    export_format,
                    export_quality,
                ],
                outputs=[
                    error_box,
                    output_image,
                    output_gallery,
                    gif_output,
                    loading_indicator,
                    status_text,
                    history_gallery,
                    download_image_button,
                    download_gif_button,
                ],
            )

            def update_export_format(mode):
                if mode == "GIF Mode":
                    return gr.update(value="GIF", interactive=False)
                else:
                    return gr.update(value="PNG", interactive=True)

            # Set up the event handler
            generation_mode.change(
                update_export_format,
                inputs=[generation_mode],
                outputs=[export_format],
            )

            # Help section with enhanced documentation

            gr.Markdown(
                """
            ### Enhanced Quick Guide

            1. **Enter a detailed prompt** describing your desired image.
            - Be as specific as possible about the subject, setting, style, and any other important details.
            - Include descriptors such as lighting, colors, mood, and composition.
            - **For Single Image Mode**, use a single, detailed prompt.
            - **For GIF Mode**, separate multiple prompts with `||` to create a sequence.
                - Example: *"A seed sprouting from the soil || A small plant emerging || The plant growing leaves || A blooming flower || The flower wilting"*

            2. **Optionally add a negative prompt** to specify unwanted elements.
            - Use this to exclude certain features or styles from the generated image.
            - Example: *"blurry, low-resolution, watermark, text"*

            3. **Choose between Single Image or GIF Mode**.
            - **Single Image**: Generates one image based on your prompt.
            - **GIF Mode**: Generates an animated GIF sequence.
                - Ensure you provide multiple prompts separated by `||` for different frames.
                - If only one prompt is provided, the system will generate multiple frames with variations based on different seeds.

            4. **Select a seed profile or use a manual seed** for reproducibility.
            - Seed profiles control the randomness and variation in image generation.
            - Using a manual seed allows you to reproduce the same image in future generations.

            5. **Configure batch processing** for multiple images.
            - **Batch Size**: Number of images to generate in one batch.
            - **Batch Type**:
                - *Same Prompt*: Generates multiple images using the same prompt.
                - *Prompt Variations*: Appends variation numbers to your prompt for each image.

            6. **Adjust advanced settings and memory optimization** if needed.
            - **Guidance Scale**: Controls how closely the image matches the prompt.
            - **Steps**: Number of denoising steps. Higher values may produce better quality but take longer.
            - **Memory Optimization**: Enable attention slicing or CPU offload to reduce memory usage.

            7. **Enable enhancements** for better quality.
            - **Enable Upscaling**: Improves image resolution.
            - **Enhance Faces**: Applies techniques to improve facial details.

            8. **Click Generate** and monitor progress.
            - A loading animation and status messages will inform you of the generation progress.

            9. **Find your generated images in the History tab**.
            - Review past generations and access image details.
            - Export or clear your history as needed.

            ### Writing Effective Prompts

            - **Be Specific**: Include as many relevant details as possible.
            - **Use Descriptive Language**: Adjectives and adverbs help to refine the image.
            - **Specify Styles**: Mention artistic styles, such as "realistic", "impressionist", "cyberpunk", etc.
            - **Include Technical Details**: Terms like "4K resolution", "high detail", "professional photography" can improve output quality.
            - **For GIF Sequences**:
            - Create a narrative or progression across your prompts.
            - Ensure each prompt logically follows the previous one for smooth animation.
            - Example: *"Empty canvas || Sketch outlines appear || Colors fill in || Details added || Finished painting"*

            ### Tips

            - Use **negative prompts** to avoid unwanted elements.
            - Experiment with different **seed profiles** for variation.
            - Use **batch processing** to generate multiple images at once.
            - Enable **memory optimizations** if you encounter performance issues.
            - **In GIF Mode**, more prompts lead to longer and smoother animations.
            - Check the **History** tab for past generations and details.
            """
            )

    return demo


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description="Flux Image Generator")

    # Add an argument to share the app publicly
    parser.add_argument("--share", action="store_true", help="Share the app publicly")

    # Add an argument to specify the port to run the app on
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")

    # Add an argument to specify the host to bind to
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Print the custom URLs
    print(f"* Running on local URL:  http://localhost:{args.port}")

    # Create and launch the interface
    demo = create_interface()
    demo.queue().launch(
        server_name=args.host,  # Use the host argument
        server_port=args.port,  # Port to run the app on
        share=args.share,  # Whether to share the app publicly
        show_error=True,  # Show errors in the interface
        quiet=True,  # Do not suppress Gradio output
    )

    # # Keep the main thread alive
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("Keyboard interruption in main thread... closing server.")
    #     demo.close()
