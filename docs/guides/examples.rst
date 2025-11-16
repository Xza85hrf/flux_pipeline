Examples
========

FluxPipeline includes comprehensive Jupyter notebooks and Python scripts.

Jupyter Notebooks
-----------------

All notebooks are located in the ``examples/`` directory.

Getting Started
~~~~~~~~~~~~~~~

1. **Basic Generation** (``01_basic_generation.ipynb``)
   
   - Initialize FluxPipeline
   - Generate your first image
   - Understanding parameters
   - Reproducible generation

2. **Batch Processing** (``02_batch_processing.ipynb``)
   
   - Generate multiple images
   - Process different prompts
   - Memory-safe batch processing
   - Export metadata

3. **GIF Creation** (``03_gif_creation.ipynb``)
   
   - Create animated GIFs
   - Day/night cycles
   - Seasonal transformations
   - Custom animation parameters

Advanced Topics
~~~~~~~~~~~~~~~

4. **Custom Prompts** (``04_custom_prompts.ipynb``)
   
   - Prompt engineering
   - Style modifiers
   - Quality boosters
   - Lighting and atmosphere

5. **Memory Optimization** (``05_memory_optimization.ipynb``)
   
   - GPU memory monitoring
   - Memory-efficient generation
   - OOM error handling
   - Performance analysis

6. **Seed Management** (``06_seed_management.ipynb``)
   
   - Understand seed profiles
   - Reproducible results
   - Seed libraries
   - Exploration techniques

Running Notebooks
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install Jupyter
   pip install jupyter ipykernel

   # Register kernel
   python -m ipykernel install --user --name=flux

   # Start Jupyter
   jupyter notebook examples/

Python Scripts
--------------

Example Scripts
~~~~~~~~~~~~~~~

**basic_example.py** - Simple generation

.. code-block:: bash

   python examples/example_basic.py

The script demonstrates:

- Environment setup
- Pipeline initialization
- Model loading
- Image generation
- Output saving

Script Template
~~~~~~~~~~~~~~~

.. code-block:: python

   #!/usr/bin/env python3
   """Custom generation script."""
   
   import sys
   from pathlib import Path
   
   # Add parent to path
   sys.path.append(str(Path(__file__).parent.parent))
   
   from pipeline import FluxPipeline
   from core import SeedProfile
   from config import setup_environment, logger
   from utils import setup_workspace
   
   def main():
       """Main generation function."""
       # Setup
       setup_environment()
       workspace = setup_workspace()
       
       # Initialize
       pipeline = FluxPipeline(workspace=workspace)
       
       if not pipeline.load_model():
           logger.error("Failed to load model")
           return 1
       
       # Your prompts
       prompts = [
           "A serene landscape",
           "A futuristic city",
           "A magical forest",
       ]
       
       # Generate
       for i, prompt in enumerate(prompts, 1):
           logger.info(f"[{i}/{len(prompts)}] {prompt}")
           
           image, seed = pipeline.generate_image(
               prompt=prompt,
               num_inference_steps=4,
               height=1024,
               width=1024,
               seed_profile=SeedProfile.BALANCED
           )
           
           if image:
               output = workspace / f"image_{i}_{seed}.png"
               image.save(output)
               logger.info(f"✅ Saved: {output}")
       
       return 0
   
   if __name__ == "__main__":
       sys.exit(main())

Common Use Cases
----------------

Portrait Generation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   prompt = """
   Professional portrait of a person, 
   studio lighting, soft focus, 
   85mm lens, shallow depth of field,
   photorealistic, highly detailed
   """
   
   image, seed = pipeline.generate_image(
       prompt=prompt,
       height=1024,
       width=768,  # Portrait aspect ratio
       num_inference_steps=4
   )

Landscape Generation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   prompt = """
   Majestic mountain landscape at golden hour,
   sweeping vista, dramatic clouds,
   professional landscape photography,
   vibrant colors, 8k resolution
   """
   
   image, seed = pipeline.generate_image(
       prompt=prompt,
       height=768,
       width=1024,  # Landscape aspect ratio
       num_inference_steps=4
   )

Concept Art
~~~~~~~~~~~

.. code-block:: python

   prompt = """
   Futuristic cyberpunk cityscape,
   neon lights, flying vehicles,
   rain-slicked streets, night scene,
   digital art, highly detailed,
   cinematic composition
   """
   
   image, seed = pipeline.generate_image(
       prompt=prompt,
       height=1024,
       width=1024,
       num_inference_steps=4,
       seed_profile=SeedProfile.CREATIVE
   )

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from datetime import datetime
   import gc
   import torch
   
   # Create batch directory
   batch_dir = workspace / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
   batch_dir.mkdir(exist_ok=True)
   
   prompts = [
       "A serene lake",
       "A busy marketplace",
       "A quiet library",
   ]
   
   for i, prompt in enumerate(prompts, 1):
       # Generate
       image, seed = pipeline.generate_image(
           prompt=prompt,
           height=768,
           width=768,
           num_inference_steps=4
       )
       
       if image:
           # Save immediately
           image.save(batch_dir / f"image_{i}_{seed}.png")
           
           # Clean up
           del image
           gc.collect()
           if torch.cuda.is_available():
               torch.cuda.empty_cache()

Tips and Tricks
---------------

1. **Start Simple**: Begin with basic prompts, then add details
2. **Use Examples**: Reference the notebooks for proven patterns
3. **Monitor Memory**: Watch GPU usage with large batches
4. **Save Seeds**: Keep track of seeds that produce good results
5. **Experiment**: Try different seed profiles and parameters
6. **Check Notebooks**: Each notebook has detailed examples and explanations

Next Steps
----------

- Download the :doc:`installation` guide
- Read :doc:`quickstart` for basics
- See :doc:`configuration` for settings
- Browse ``examples/`` directory in the repository
