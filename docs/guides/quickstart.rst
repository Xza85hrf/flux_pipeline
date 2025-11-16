Quick Start
===========

This guide will get you generating images in 5 minutes.

Basic Usage
-----------

1. Setup and Initialize
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pipeline import FluxPipeline
   from core import SeedProfile
   from config import setup_environment
   from utils import setup_workspace

   # Setup environment
   setup_environment()
   workspace = setup_workspace()

   # Initialize pipeline
   pipeline = FluxPipeline(workspace=workspace)

2. Load Model
~~~~~~~~~~~~~

.. code-block:: python

   # Load FLUX.1-schnell model (first run will download ~24GB)
   if pipeline.load_model():
       print("✅ Model loaded successfully!")
   else:
       print("❌ Failed to load model")

3. Generate Your First Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate image
   prompt = "A serene mountain landscape at sunset, photorealistic"
   
   image, seed = pipeline.generate_image(
       prompt=prompt,
       num_inference_steps=4,      # FLUX.1-schnell works best with 1-4 steps
       guidance_scale=0.0,          # Recommended for FLUX.1-schnell
       height=1024,
       width=1024,
       seed_profile=SeedProfile.BALANCED
   )

   # Save image
   if image:
       output_path = workspace / f"my_image_{seed}.png"
       image.save(output_path)
       print(f"✅ Image saved to: {output_path}")
       print(f"   Seed: {seed}")

Complete Example
----------------

.. code-block:: python

   #!/usr/bin/env python3
   """Simple image generation script."""
   
   from pipeline import FluxPipeline
   from core import SeedProfile
   from config import setup_environment, logger
   from utils import setup_workspace

   def main():
       # Setup
       logger.info("Initializing FluxPipeline...")
       setup_environment()
       workspace = setup_workspace()

       # Initialize and load
       pipeline = FluxPipeline(workspace=workspace)
       
       if not pipeline.load_model():
           logger.error("Failed to load model")
           return 1

       # Generate
       prompt = "A magical forest with glowing mushrooms and fireflies"
       logger.info(f"Generating: {prompt}")

       image, seed = pipeline.generate_image(
           prompt=prompt,
           num_inference_steps=4,
           height=1024,
           width=1024,
           seed_profile=SeedProfile.BALANCED
       )

       # Save
       if image:
           output_path = workspace / f"output_{seed}.png"
           image.save(output_path)
           logger.info(f"✅ Saved to: {output_path}")
           return 0
       else:
           logger.error("Generation failed")
           return 1

   if __name__ == "__main__":
       exit(main())

GUI Mode
--------

Launch the Gradio web interface:

.. code-block:: bash

   python gui.py --port 7860

Then open http://localhost:7860 in your browser.

Interactive Mode
----------------

Use the interactive CLI:

.. code-block:: bash

   python interactive_generation.py

Common Parameters
-----------------

Prompt
~~~~~~

The text description of what you want to generate.

**Tips:**

- Be specific and descriptive
- Include style keywords (e.g., "photorealistic", "oil painting")
- Add lighting/atmosphere (e.g., "golden hour", "soft lighting")
- Mention quality (e.g., "highly detailed", "8k resolution")

Num Inference Steps
~~~~~~~~~~~~~~~~~~~

Number of denoising steps. FLUX.1-schnell is optimized for low steps.

- **1-4 steps**: Recommended for FLUX.1-schnell (fast, good quality)
- **4-8 steps**: Higher quality but slower
- **>8 steps**: Usually no improvement for schnell model

Guidance Scale
~~~~~~~~~~~~~~

How closely to follow the prompt. For FLUX.1-schnell:

- **0.0**: Recommended (model is trained without guidance)
- **>0**: May reduce quality for schnell variant

Image Size
~~~~~~~~~~

Resolution in pixels (must be divisible by 8):

- **512x512**: Fast, low memory
- **768x768**: Balanced
- **1024x1024**: High quality (default)
- **1536x1536+**: Very high quality, requires lots of VRAM

Seed Profiles
~~~~~~~~~~~~~

Control variation level:

- ``SeedProfile.CONSERVATIVE``: Low variation (seeds 42-9999)
- ``SeedProfile.BALANCED``: Medium variation (default)
- ``SeedProfile.CREATIVE``: High variation

Or use a fixed seed for reproducibility:

.. code-block:: python

   image, seed = pipeline.generate_image(
       prompt=prompt,
       seed=42,  # Always generates same image
       ...
   )

Next Steps
----------

- Check out :doc:`examples` for Jupyter notebooks
- Learn :doc:`configuration` options
- See :doc:`../api/pipeline` for full API reference
- Browse example scripts in ``examples/`` directory
