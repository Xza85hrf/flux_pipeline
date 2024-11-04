"""Main entry point for the FluxPipeline image generation system.

This module provides the main execution flow for the FluxPipeline project,
demonstrating:
- Environment setup and configuration
- Pipeline initialization
- Image generation with different seed profiles
- Error handling and logging

The module showcases various generation scenarios including:
- Generation with different seed profiles
- Generation with manual seeds
- Generation with custom parameters

Example:
    Run the application:
    ```bash
    python main.py
    ```

Note:
    The module automatically handles environment setup, warning suppression,
    and proper error handling for production use.
"""

from config.logging_config import logger  # Fix incorrect import from venv
import warnings
from pathlib import Path
from config.env_config import setup_environment
from utils.system_utils import setup_workspace, suppress_warnings, setup_nltk
from pipeline.flux_pipeline import FluxPipeline
from core.seed_manager import SeedProfile


def main():
    """Main execution function for the FluxPipeline system.

    This function:
    1. Sets up the environment and workspace
    2. Initializes the generation pipeline
    3. Demonstrates different generation scenarios
    4. Handles errors and cleanup

    The function showcases various generation capabilities including:
    - Different seed profiles (Conservative, Balanced, Experimental)
    - Manual seed specification
    - Custom generation parameters

    Example:
        ```python
        # Run the main function
        if __name__ == "__main__":
            main()
        ```

    Note:
        The function includes proper error handling and logging
        for production use.
    """
    # Initialize environment and workspace
    setup_environment()  # Configure environment variables
    suppress_warnings()  # Suppress non-critical warnings
    setup_nltk()  # Initialize NLTK for text processing
    workspace = setup_workspace()  # Setup directory structure

    # Initialize generation pipeline
    logger.info("Initializing FluxPipeline...")
    pipeline = FluxPipeline(workspace=workspace)

    # Load model with error handling
    if not pipeline.load_model():
        logger.error("Failed to load the model. Exiting.")
        return

    # Example generation prompt
    prompt = (
        "High-quality portrait of a beautiful gothic woman with a voluptuous figure, "
        "pale skin, dark makeup, and long black hair. She is wearing elegant gothic attire "
        "with intricate lace and accessories. The setting has a moody, atmospheric background "
        "with dramatic lighting to enhance the overall aesthetic. Realistic, highly detailed, "
        "professional photography, 4k resolution, masterpiece."
    )

    # Demonstrate generation with different seed profiles
    logger.info("Generating images with different seed profiles...")
    for profile in [
        SeedProfile.CONSERVATIVE,  # Lower range seeds for stability
        SeedProfile.BALANCED,  # Mid-range seeds for general use
        SeedProfile.EXPERIMENTAL,  # High range seeds for variation
    ]:
        logger.info(f"\nGenerating with {profile.value} profile")
        pipeline.generate_image(
            prompt=prompt,
            profile=profile,
            guidance_scale=0.0,  # No guidance scaling
            num_inference_steps=4,  # Fast generation
            height=1024,  # High resolution output
            width=1024,
        )

    # Demonstrate generation with manual seed
    logger.info("\nGenerating with manual seed...")
    pipeline.generate_image(
        prompt=prompt,
        profile=SeedProfile.BALANCED,
        manual_seed=83,  # Specific seed for reproducibility
        guidance_scale=0.0,
        num_inference_steps=4,
        height=1024,
        width=1024,
    )


if __name__ == "__main__":
    # Suppress warnings during execution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            # Run main function
            main()
        except KeyboardInterrupt:
            # Handle user interruption gracefully
            logger.info("Generation interrupted by user")
        except Exception as e:
            # Log unexpected errors with full traceback
            logger.error(f"Unexpected error: {str(e)}")
            import traceback

            traceback.print_exc()
