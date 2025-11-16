Architecture
============

This document describes the architecture and design of FluxPipeline.

System Overview
---------------

FluxPipeline is organized into modular components:

::

   FluxPipeline
   ├── Pipeline Layer        # High-level generation interface
   ├── Core Layer           # GPU, memory, seed, prompt management
   ├── Config Layer         # Environment, logging, validation
   ├── Utils Layer          # Utilities, metrics, system tools
   └── API Layer            # Health checks, monitoring

Component Diagram
~~~~~~~~~~~~~~~~~

::

   ┌─────────────────────────────────────────────────────────┐
   │                     User Interface                       │
   │         (GUI / CLI / Notebooks / API)                   │
   └──────────────────────┬──────────────────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────────────────┐
   │                  Pipeline Layer                          │
   │                 (FluxPipeline)                          │
   │  - Orchestrates generation workflow                     │
   │  - Manages model loading                                │
   │  - Handles image generation                             │
   └──────────────────────┬──────────────────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           │              │               │
   ┌───────▼────┐ ┌──────▼─────┐ ┌──────▼──────┐
   │ Core Layer │ │Config Layer│ │ Utils Layer │
   ├────────────┤ ├────────────┤ ├─────────────┤
   │GPU Manager │ │Environment │ │ Logging     │
   │Memory Mgr  │ │Logging     │ │ Metrics     │
   │Seed Manager│ │Validation  │ │ System      │
   │Prompt Mgr  │ └────────────┘ └─────────────┘
   └────────────┘
           │
   ┌───────▼────────────────┐
   │   External Libraries    │
   │  (PyTorch, Diffusers)  │
   └────────────────────────┘

Module Details
--------------

Pipeline Module
~~~~~~~~~~~~~~~

**Location**: ``pipeline/flux_pipeline.py``

**Responsibilities**:

- Model loading and initialization
- Image generation orchestration
- Parameter validation
- Resource management

**Key Classes**:

- ``FluxPipeline``: Main pipeline class

**Dependencies**:

- ``diffusers.FluxPipeline``
- ``torch``
- Core managers

Core Module
~~~~~~~~~~~

GPU Manager
^^^^^^^^^^^

**Location**: ``core/gpu_manager.py``

**Responsibilities**:

- Detect available GPUs (NVIDIA, AMD, Intel)
- Select best GPU for generation
- Monitor GPU status
- Handle multi-GPU scenarios

**Key Classes**:

- ``MultiGPUManager``: GPU detection and selection
- ``GPUVendor``: Enum for GPU vendors
- ``GPUInfo``: GPU information dataclass

Memory Manager
^^^^^^^^^^^^^^

**Location**: ``core/memory_manager.py``

**Responsibilities**:

- Monitor memory usage (CPU and GPU)
- Detect memory pressure
- Clear caches when needed
- Prevent OOM errors

**Key Classes**:

- ``MemoryManager``: Memory monitoring and management

Seed Manager
^^^^^^^^^^^^

**Location**: ``core/seed_manager.py``

**Responsibilities**:

- Generate seeds with different profiles
- Ensure reproducibility
- Manage seed statistics

**Key Classes**:

- ``SeedManager``: Seed generation and management
- ``SeedProfile``: Enum for seed profiles (CONSERVATIVE, BALANCED, CREATIVE)

Prompt Manager
^^^^^^^^^^^^^^

**Location**: ``core/prompt_manager.py``

**Responsibilities**:

- Validate prompts
- Enhance prompts with keywords
- Manage prompt history

**Key Classes**:

- ``PromptManager``: Prompt handling

Config Module
~~~~~~~~~~~~~

**Location**: ``config/``

**Responsibilities**:

- Environment configuration
- Logging setup
- Configuration validation

**Key Files**:

- ``env_config.py``: Environment detection and setup
- ``logging_config.py``: Logging configuration
- ``validator.py``: Configuration validation

Utils Module
~~~~~~~~~~~~

**Location**: ``utils/``

**Responsibilities**:

- Performance logging
- System utilities
- Performance metrics collection

**Key Files**:

- ``logging_utils.py``: Performance logging decorators
- ``system_utils.py``: Workspace setup, warnings suppression
- ``metrics.py``: Performance metrics collection and analysis

API Module
~~~~~~~~~~

**Location**: ``api/``

**Responsibilities**:

- Health check endpoints
- Monitoring integration
- Kubernetes probes

**Key Files**:

- ``health.py``: Health check implementations

Data Flow
---------

Image Generation Flow
~~~~~~~~~~~~~~~~~~~~~

::

   1. User Request
      ↓
   2. FluxPipeline.generate_image()
      ↓
   3. Parameter Validation (ConfigValidator)
      ↓
   4. Seed Generation (SeedManager)
      ↓
   5. Prompt Processing (PromptManager)
      ↓
   6. Memory Check (MemoryManager)
      ↓
   7. GPU Selection (MultiGPUManager)
      ↓
   8. Model Inference (diffusers.FluxPipeline)
      ↓
   9. Image Post-processing
      ↓
   10. Return Image + Seed

Model Loading Flow
~~~~~~~~~~~~~~~~~~

::

   1. FluxPipeline.load_model()
      ↓
   2. Environment Setup (setup_environment)
      ↓
   3. GPU Detection (MultiGPUManager)
      ↓
   4. Memory Check (MemoryManager)
      ↓
   5. Model Download (HuggingFace)
      ↓
   6. Model Loading (diffusers)
      ↓
   7. Model to GPU
      ↓
   8. Ready for Inference

Configuration Flow
~~~~~~~~~~~~~~~~~~

::

   1. Load .env file
      ↓
   2. Parse environment variables
      ↓
   3. Validate configuration (ConfigValidator)
      ↓
   4. Setup logging (setup_logging)
      ↓
   5. Configure GPU (MultiGPUManager)
      ↓
   6. Apply settings

Design Patterns
---------------

Singleton Pattern
~~~~~~~~~~~~~~~~~

Used for:

- Logger instance (``config.logger``)
- Global metrics (``utils.get_global_metrics()``)

**Benefits**: Single source of truth, shared state

Manager Pattern
~~~~~~~~~~~~~~~

Used for:

- ``MultiGPUManager``
- ``MemoryManager``
- ``SeedManager``
- ``PromptManager``

**Benefits**: Encapsulation, single responsibility

Context Manager Pattern
~~~~~~~~~~~~~~~~~~~~~~~

Used for:

- ``PerformanceMetrics.measure()``
- ``suppress_warnings()``

**Benefits**: Resource cleanup, RAII

Factory Pattern
~~~~~~~~~~~~~~~

Used for:

- Seed generation based on profile
- GPU selection based on vendor

**Benefits**: Flexibility, extensibility

Error Handling
--------------

Error Hierarchy
~~~~~~~~~~~~~~~

::

   Exception
   ├── RuntimeError
   │   ├── GPU not available
   │   ├── Out of memory
   │   └── Model load failed
   ├── ValueError
   │   ├── Invalid configuration
   │   └── Invalid parameters
   └── ImportError
       └── Missing dependencies

Error Handling Strategy
~~~~~~~~~~~~~~~~~~~~~~~

1. **Validate Early**: Check parameters before expensive operations
2. **Fail Fast**: Return errors quickly instead of continuing
3. **Log Errors**: Always log errors with context
4. **User-Friendly Messages**: Provide actionable error messages
5. **Graceful Degradation**: Fall back to CPU if GPU fails

Performance Considerations
--------------------------

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~

1. **Memory Management**:
   - Clear cache between generations
   - Use attention slicing for low VRAM
   - Monitor memory pressure

2. **Model Loading**:
   - Load model once, reuse for multiple generations
   - Cache model on GPU
   - Use appropriate precision (float16 vs float32)

3. **Batch Processing**:
   - Process images sequentially to avoid OOM
   - Clear memory between batches
   - Use memory-efficient data structures

4. **GPU Utilization**:
   - Select best GPU automatically
   - Monitor GPU utilization
   - Avoid unnecessary CPU<->GPU transfers

Monitoring and Metrics
~~~~~~~~~~~~~~~~~~~~~~

- **PerformanceMetrics**: Track timing and memory
- **Logger**: Record events and errors
- **Health Checks**: Monitor system status

Extensibility
-------------

Adding New GPU Vendors
~~~~~~~~~~~~~~~~~~~~~~

1. Add vendor to ``GPUVendor`` enum
2. Implement detection in ``MultiGPUManager.detect_vendor()``
3. Add device setup logic
4. Update documentation

Adding New Seed Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add profile to ``SeedProfile`` enum
2. Implement range in ``SeedManager.get_seed()``
3. Update tests
4. Document new profile

Adding New Health Checks
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add function to ``api/health.py``
2. Export in ``api/__init__.py``
3. Add FastAPI route if needed
4. Document usage

Testing Strategy
----------------

See :doc:`testing` for detailed testing information.

**Test Pyramid**:

::

   ┌──────────────┐
   │ Integration  │  (Few, slow, full stack)
   ├──────────────┤
   │     Unit     │  (Many, fast, isolated)
   └──────────────┘

Future Enhancements
-------------------

Potential improvements:

1. **Model Support**: Add other diffusion models
2. **Image-to-Image**: Support img2img generation
3. **Inpainting**: Add inpainting capabilities
4. **LoRA Support**: Fine-tune with LoRA adapters
5. **API Server**: FastAPI REST API
6. **WebUI Improvements**: Enhanced Gradio interface
7. **Cloud Integration**: S3, Azure Blob, GCS support
8. **Distributed Generation**: Multi-node support

Contributing
------------

See :doc:`contributing` for contribution guidelines.

References
----------

- **Diffusers**: https://huggingface.co/docs/diffusers
- **FLUX.1**: https://blackforestlabs.ai/flux-1/
- **PyTorch**: https://pytorch.org/docs/stable/
