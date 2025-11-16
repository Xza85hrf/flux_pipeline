Core Module
===========

The core module contains essential components for GPU management, memory optimization,
prompt handling, and seed control.

GPU Manager
-----------

.. automodule:: core.gpu_manager
   :members:
   :undoc-members:
   :show-inheritance:

MultiGPUManager
~~~~~~~~~~~~~~~

.. autoclass:: core.gpu_manager.MultiGPUManager
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Memory Manager
--------------

.. automodule:: core.memory_manager
   :members:
   :undoc-members:
   :show-inheritance:

MemoryManager
~~~~~~~~~~~~~

.. autoclass:: core.memory_manager.MemoryManager
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Prompt Manager
--------------

.. automodule:: core.prompt_manager
   :members:
   :undoc-members:
   :show-inheritance:

PromptManager
~~~~~~~~~~~~~

.. autoclass:: core.prompt_manager.PromptManager
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Seed Manager
------------

.. automodule:: core.seed_manager
   :members:
   :undoc-members:
   :show-inheritance:

SeedManager
~~~~~~~~~~~

.. autoclass:: core.seed_manager.SeedManager
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

SeedProfile
~~~~~~~~~~~

.. autoclass:: core.seed_manager.SeedProfile
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

GPU Management
~~~~~~~~~~~~~~

.. code-block:: python

   from core import MultiGPUManager

   # Detect GPUs
   gpu_manager = MultiGPUManager()
   print(f"Detected {len(gpu_manager.available_gpus)} GPUs")

   # Get best GPU
   best_gpu = gpu_manager.get_best_gpu()
   print(f"Using GPU {best_gpu.device_id}: {best_gpu.name}")

Memory Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from core import MemoryManager

   # Initialize memory manager
   memory_manager = MemoryManager()

   # Check memory status
   status = memory_manager.get_memory_status()
   print(f"GPU Memory: {status['gpu_memory_used_percent']:.1f}%")

   # Clear cache if needed
   if status['gpu_memory_used_percent'] > 80:
       memory_manager.clear_cache()

Seed Control
~~~~~~~~~~~~

.. code-block:: python

   from core import SeedManager, SeedProfile

   # Get seed with different profiles
   seed_manager = SeedManager()

   # Conservative (low variation)
   seed1 = seed_manager.get_seed(SeedProfile.CONSERVATIVE)

   # Balanced (medium variation)
   seed2 = seed_manager.get_seed(SeedProfile.BALANCED)

   # Creative (high variation)
   seed3 = seed_manager.get_seed(SeedProfile.CREATIVE)

   # Fixed seed for reproducibility
   seed4 = seed_manager.get_seed(seed=42)
