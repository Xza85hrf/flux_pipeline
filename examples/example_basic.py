#!/usr/bin/env python3
"""Basic image generation example script.

This script demonstrates the simplest way to generate an image with FluxPipeline.

Usage:
    python examples/example_basic.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline import FluxPipeline
from core import SeedProfile
from config import setup_environment, logger
from utils import setup_workspace


def main():
    """Generate a basic image."""
    logger.info("Starting basic image generation example")

    # Setup
    setup_environment()
    workspace = setup_workspace()

    # Initialize pipeline
    pipeline = FluxPipeline(workspace=workspace)

    # Load model
    if not pipeline.load_model():
        logger.error("Failed to load model")
        return 1

    # Generate image
    prompt = "A serene mountain landscape at sunset with crystal clear lake"
    logger.info(f"Generating image with prompt: {prompt}")

    image, seed = pipeline.generate_image(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=1024,
        width=1024,
        seed_profile=SeedProfile.BALANCED
    )

    if image:
        # Save the image
        output_path = workspace / f"basic_example_{seed}.png"
        image.save(output_path)
        logger.info(f"✅ Image saved to: {output_path}")
        logger.info(f"Used seed: {seed}")
        return 0
    else:
        logger.error("❌ Image generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
