"""
Generate a transformation GIF using the FluxPipeline.

This script generates a series of images based on a list of prompts and creates a GIF that shows a transformation from one image to another. The script uses the FluxPipeline to generate the images and the PIL library to create the GIF.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from PIL import Image
from pathlib import Path
from typing import List

from config.env_config import setup_environment
from pipeline.flux_pipeline import FluxPipeline
from core.seed_manager import SeedProfile
from utils.system_utils import setup_workspace, suppress_warnings
from config.logging_config import logger

# Suppress warnings and setup environment
suppress_warnings()
setup_environment()

# Initialize workspace and pipeline
workspace = setup_workspace()
pipeline = FluxPipeline(workspace=workspace)

# Load the model
if not pipeline.load_model():
    logger.error("Failed to load the model. Exiting.")
    exit(1)


def generate_transformation_images(
    prompts: List[str],
    output_dir: Path,
    num_inference_steps: int = 4,
    guidance_scale: float = 0.0,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
) -> List[Image.Image]:
    """
    Generate a series of images based on a list of prompts.

    Args:
        prompts: A list of prompts to generate images from.
        output_dir: The directory to save the generated images.
        num_inference_steps: The number of inference steps to use.
        guidance_scale: The guidance scale to use.
        height: The height of the generated images.
        width: The width of the generated images.
        seed: The seed to use for image generation.

    Returns:
        A list of generated images.
    """
    images = []
    for idx, prompt in enumerate(prompts):
        logger.info(f"Generating image {idx + 1}/{len(prompts)}")
        image, used_seed = pipeline.generate_image(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
        )
        if image:
            # Save the image
            image_path = output_dir / f"frame_{idx:03d}.png"
            image.save(image_path)
            images.append(image)
        else:
            logger.error(f"Failed to generate image for prompt: {prompt}")
    return images


def create_gif(images: List[Image.Image], output_path: Path, duration: int = 500):
    """
    Create a GIF from a list of images.

    Args:
        images: A list of PIL Image objects.
        output_path: The path to save the GIF.
        duration: The duration of each frame in the GIF.
    """
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    logger.info(f"GIF saved to {output_path}")


def main():
    """
    Main function to generate a transformation GIF.

    This function defines the transformation prompts, generates the images, and creates a GIF from the generated images.
    """
    # Define the transformation prompts
    prompts = [
        "A high-quality photo of a cute cat running in a grassy field, ultra-realistic, 4k resolution.",
        "A photo of a cat running, starting to morph into a giant creature, detailed, high-resolution.",
        "An image of a cat gradually transforming into Godzilla, mid-transformation, photorealistic.",
        "A depiction of Godzilla emerging from the form of a cat, highly detailed, dramatic lighting.",
        "A high-quality photo of Godzilla roaring in a city, ultra-realistic, 4k resolution.",
    ]

    output_dir = Path("transformation_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate images
    images = generate_transformation_images(
        prompts=prompts,
        output_dir=output_dir,
        num_inference_steps=10,  # Increase for better quality
        guidance_scale=3.5,  # Standard value for Stable Diffusion models
        height=512,
        width=512,
        seed=42,  # Use a fixed seed for consistency
    )

    if images:
        # Create GIF
        output_gif_path = output_dir / "cat_to_godzilla.gif"
        create_gif(images, output_gif_path, duration=500)
    else:
        logger.error("No images were generated. Cannot create GIF.")


if __name__ == "__main__":
    main()
