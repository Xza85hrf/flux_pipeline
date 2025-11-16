Configuration
=============

FluxPipeline can be configured through environment variables, configuration files,
and programmatic settings.

Environment Variables
---------------------

GPU Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Select specific GPU(s)
   export CUDA_VISIBLE_DEVICES=0,1

   # Choose GPU vendor
   export GPU_VENDOR=cuda  # or rocm, intel, cpu

   # Memory management
   export MEMORY_THRESHOLD=0.9
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Model name
   export MODEL_NAME=black-forest-labs/FLUX.1-schnell

   # Precision
   export TORCH_DTYPE=float16  # or float32, bfloat16

   # Cache directory
   export HF_HOME=~/.cache/huggingface

Workspace Settings
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Workspace directory
   export WORKSPACE_DIR=./workspace

   # Output directory
   export OUTPUT_DIR=./outputs

Logging
~~~~~~~

.. code-block:: bash

   # Log level
   export LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

   # Log file
   export LOG_FILE=flux_pipeline.log

HuggingFace
~~~~~~~~~~~

.. code-block:: bash

   # Authentication token (for private models)
   export HF_TOKEN=your_token_here
   
   # Or
   export HUGGING_FACE_HUB_TOKEN=your_token_here

Configuration File
------------------

Create a ``.env`` file in the project root:

.. code-block:: bash

   # GPU
   CUDA_VISIBLE_DEVICES=0
   GPU_VENDOR=cuda
   MEMORY_THRESHOLD=0.9

   # Model
   MODEL_NAME=black-forest-labs/FLUX.1-schnell
   TORCH_DTYPE=float16

   # Workspace
   WORKSPACE_DIR=./workspace
   OUTPUT_DIR=./outputs

   # Logging
   LOG_LEVEL=INFO
   LOG_FILE=flux_pipeline.log

   # HuggingFace
   HF_TOKEN=your_token_here

Programmatic Configuration
---------------------------

Default Model Config
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from config import DEFAULT_MODEL_CONFIG

   # View default configuration
   print(DEFAULT_MODEL_CONFIG)

   # Modify configuration
   custom_config = DEFAULT_MODEL_CONFIG.copy()
   custom_config['torch_dtype'] = 'float32'
   custom_config['attention_slicing'] = True

Default Generation Config
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from config import DEFAULT_GENERATION_CONFIG

   # View defaults
   print(DEFAULT_GENERATION_CONFIG)

   # Customize
   custom_gen_config = DEFAULT_GENERATION_CONFIG.copy()
   custom_gen_config['num_inference_steps'] = 8
   custom_gen_config['height'] = 1536
   custom_gen_config['width'] = 1536

Validation
~~~~~~~~~~

.. code-block:: python

   from config import ConfigValidator, validate_and_log

   # Validate configuration
   config = {
       'memory_threshold': 0.9,
       'num_inference_steps': 4,
       'height': 1024,
       'width': 1024
   }

   # Quick validation
   validate_and_log(config, raise_on_error=True)

   # Manual validation
   validator = ConfigValidator()
   errors = validator.validate_model_config(config)
   
   if errors:
       print("Configuration errors:")
       for error in errors:
           print(f"  - {error}")

Hardware-Specific Settings
---------------------------

NVIDIA CUDA
~~~~~~~~~~~

.. code-block:: bash

   # Environment
   export GPU_VENDOR=cuda
   export CUDA_VISIBLE_DEVICES=0

   # Memory optimization
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

.. code-block:: python

   # In code
   pipeline = FluxPipeline(
       workspace=workspace,
       torch_dtype='float16',  # Use FP16 for faster inference
       attention_slicing=True   # Enable for lower VRAM usage
   )

AMD ROCm
~~~~~~~~

.. code-block:: bash

   # Environment
   export GPU_VENDOR=rocm
   export ROCR_VISIBLE_DEVICES=0

Intel GPUs
~~~~~~~~~~

.. code-block:: bash

   # Environment
   export GPU_VENDOR=intel
   export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

CPU Only
~~~~~~~~

.. code-block:: bash

   # Environment
   export GPU_VENDOR=cpu

.. code-block:: python

   # In code - use smaller dimensions
   image, seed = pipeline.generate_image(
       prompt=prompt,
       height=512,
       width=512,
       num_inference_steps=4
   )

Memory Optimization
-------------------

For Low VRAM GPUs
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Enable attention slicing
   pipeline = FluxPipeline(attention_slicing=True)

   # Use lower resolution
   image, seed = pipeline.generate_image(
       prompt=prompt,
       height=512,
       width=512
   )

   # Clear cache between generations
   import torch
   torch.cuda.empty_cache()

For High VRAM GPUs
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Disable attention slicing for speed
   pipeline = FluxPipeline(attention_slicing=False)

   # Use high resolution
   image, seed = pipeline.generate_image(
       prompt=prompt,
       height=1536,
       width=1536
   )

Docker Configuration
--------------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker run --gpus all \
       -e CUDA_VISIBLE_DEVICES=0 \
       -e MODEL_NAME=black-forest-labs/FLUX.1-schnell \
       -e TORCH_DTYPE=float16 \
       -e LOG_LEVEL=INFO \
       -p 7860:7860 \
       fluxpipeline:cuda

Volume Mounts
~~~~~~~~~~~~~

.. code-block:: bash

   docker run --gpus all \
       -v $(pwd)/workspace:/app/workspace \
       -v $(pwd)/.env:/app/.env \
       -v ~/.cache/huggingface:/root/.cache/huggingface \
       -p 7860:7860 \
       fluxpipeline:cuda

Docker Compose
~~~~~~~~~~~~~~

.. code-block:: yaml

   version: '3.8'
   services:
     flux:
       image: fluxpipeline:cuda
       environment:
         - CUDA_VISIBLE_DEVICES=0
         - TORCH_DTYPE=float16
         - LOG_LEVEL=INFO
       volumes:
         - ./workspace:/app/workspace
         - ./.env:/app/.env
       ports:
         - "7860:7860"
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]

Best Practices
--------------

1. **Use .env file** for local development
2. **Validate configuration** before starting long runs
3. **Monitor memory** usage and adjust accordingly
4. **Set HF_TOKEN** in environment, not in code
5. **Use appropriate precision** (float16 for most cases)
6. **Enable attention slicing** if you have <16GB VRAM
7. **Clear cache** between generations in batch mode

Next Steps
----------

- See :doc:`examples` for usage patterns
- Check :doc:`../api/config` for API reference
- Review :doc:`installation` for setup details
