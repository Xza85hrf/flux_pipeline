Installation
============

This guide covers different installation methods for FluxPipeline.

Requirements
------------

- Python 3.10+ (3.11-3.13 recommended)
- CUDA 12.4+ (for NVIDIA GPUs)
- 16GB+ GPU memory recommended
- 32GB+ system RAM recommended

Method 1: Conda (Recommended)
------------------------------

.. code-block:: bash

   # Create conda environment
   conda create -n flux python=3.12
   conda activate flux

   # Clone repository
   git clone https://github.com/Xza85hrf/flux_pipeline.git
   cd flux_pipeline

   # Install dependencies (choose one)
   # For NVIDIA CUDA:
   pip install -r requirements_cuda.txt

   # For AMD ROCm:
   pip install -r requirements_rocm.txt

   # For Intel GPUs:
   pip install -r requirements_intel.txt

   # For CPU only:
   pip install -r requirements_cpu.txt

Method 2: venv
--------------

.. code-block:: bash

   # Create virtual environment
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Clone and install
   git clone https://github.com/Xza85hrf/flux_pipeline.git
   cd flux_pipeline
   pip install -r requirements_cuda.txt

Method 3: Docker
----------------

CUDA Support
~~~~~~~~~~~~

.. code-block:: bash

   # Build
   docker build -f Dockerfile.cuda -t fluxpipeline:cuda .

   # Run
   docker run --gpus all -p 7860:7860 fluxpipeline:cuda

CPU Only
~~~~~~~~

.. code-block:: bash

   # Build
   docker build -f Dockerfile.cpu -t fluxpipeline:cpu .

   # Run
   docker run -p 7860:7860 fluxpipeline:cpu

Method 4: Docker Compose
------------------------

.. code-block:: bash

   # Start with CUDA
   docker-compose up flux-cuda

   # Or start with CPU
   docker-compose up flux-cpu

   # Or development mode
   docker-compose up flux-dev

Development Installation
------------------------

For development with all tools:

.. code-block:: bash

   # Install with dev dependencies
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

   # Run tests
   pytest tests/ -v

   # Run linting
   make lint

Verify Installation
-------------------

.. code-block:: python

   import torch
   from pipeline import FluxPipeline
   from config import setup_environment

   # Check PyTorch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")

   # Setup environment
   setup_environment()

   # Initialize pipeline
   pipeline = FluxPipeline()
   print("✅ Installation successful!")

Environment Variables
---------------------

Create a ``.env`` file (optional):

.. code-block:: bash

   # Copy example
   cp .env.example .env

   # Edit configuration
   # GPU settings
   CUDA_VISIBLE_DEVICES=0
   GPU_VENDOR=cuda

   # Model settings
   MODEL_NAME=black-forest-labs/FLUX.1-schnell
   TORCH_DTYPE=float16

   # Workspace
   WORKSPACE_DIR=./workspace

   # HuggingFace token (if needed)
   HF_TOKEN=your_token_here

Troubleshooting
---------------

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Reduce image size or enable attention slicing
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

Import Errors
~~~~~~~~~~~~~

.. code-block:: bash

   # Ensure you're in the correct environment
   conda activate flux  # or: source venv/bin/activate

   # Reinstall dependencies
   pip install -r requirements_cuda.txt

Model Download Issues
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Set HuggingFace token
   export HF_TOKEN=your_token_here

   # Or add to .env file
   echo "HF_TOKEN=your_token_here" >> .env

Next Steps
----------

- :doc:`quickstart` - Get started with basic usage
- :doc:`configuration` - Configure FluxPipeline
- :doc:`examples` - See example notebooks
