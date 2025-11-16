FluxPipeline Documentation
==========================

**FluxPipeline** is a professional AI image generation framework using FLUX.1-schnell.

Features
--------

- 🚀 Multi-GPU support (NVIDIA CUDA, AMD ROCm, Intel OneAPI)
- 🎨 Advanced seed management for reproducible generation
- 💾 Intelligent memory management
- 📊 Performance monitoring and metrics
- 🔍 Configuration validation
- 🐳 Docker support with multiple variants
- 📓 Complete Jupyter notebook tutorials
- 🧪 Comprehensive test suite

Quick Start
-----------

.. code-block:: python

   from pipeline import FluxPipeline
   from core import SeedProfile
   from config import setup_environment
   from utils import setup_workspace

   # Setup
   setup_environment()
   workspace = setup_workspace()

   # Initialize pipeline
   pipeline = FluxPipeline(workspace=workspace)
   pipeline.load_model()

   # Generate image
   image, seed = pipeline.generate_image(
       prompt="A serene mountain landscape at sunset",
       num_inference_steps=4,
       height=1024,
       width=1024,
       seed_profile=SeedProfile.BALANCED
   )

   # Save
   image.save("output.png")

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/installation
   guides/quickstart
   guides/configuration
   guides/examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/pipeline
   api/core
   api/config
   api/utils
   api/api_module

.. toctree::
   :maxdepth: 1
   :caption: Development

   guides/contributing
   guides/testing
   guides/architecture

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
