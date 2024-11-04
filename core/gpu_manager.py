"""GPU Management module for multi-vendor GPU support in the FluxPipeline project.

This module provides comprehensive GPU management capabilities across different vendors
(NVIDIA, AMD, Intel) with features including:
- Automatic GPU detection and initialization
- Memory management and optimization
- Model distribution across multiple GPUs
- Vendor-specific optimizations and fallbacks
- Real-time GPU statistics monitoring

Example:
    Basic usage:
    ```python
    from core.gpu_manager import MultiGPUManager
    
    # Initialize GPU manager
    gpu_manager = MultiGPUManager()
    
    # Get optimal devices for computation
    devices = gpu_manager.get_optimal_devices()
    
    # Distribute a model across available GPUs
    distributed_model = gpu_manager.distribute_model(model)
    
    # Get memory statistics
    stats = gpu_manager.get_memory_stats()
    ```

Note:
    This module automatically handles vendor-specific configurations and provides
    appropriate fallbacks when certain GPU types are unavailable.
"""

import torch
import platform
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from config.logging_config import logger


class GPUVendor(Enum):
    """Enumeration of supported GPU vendors.
    
    This enum provides standardized identification of GPU vendors across the system,
    enabling vendor-specific optimizations and configurations.

    Attributes:
        NVIDIA (str): NVIDIA GPUs (CUDA-enabled devices)
        AMD (str): AMD GPUs (ROCm/HIP-enabled devices)
        INTEL (str): Intel GPUs (OneAPI/IPEX-enabled devices)
        UNKNOWN (str): Unidentified GPU vendors
    """
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """Data structure for storing comprehensive GPU information.
    
    This class encapsulates all relevant information about a GPU device,
    including hardware specifications and runtime metrics.

    Attributes:
        index (int): Zero-based device index
        name (str): GPU model name/identifier
        vendor (GPUVendor): GPU manufacturer
        total_memory (int): Total GPU memory in bytes
        available_memory (int): Currently available memory in bytes
        compute_capability (Optional[tuple]): CUDA compute capability version (major, minor)
        temperature (Optional[int]): Current GPU temperature in Celsius
        utilization (Optional[int]): Current GPU utilization percentage

    Example:
        ```python
        gpu_info = GPUInfo(
            index=0,
            name="NVIDIA GeForce RTX 3080",
            vendor=GPUVendor.NVIDIA,
            total_memory=10_737_418_240,  # 10GB
            available_memory=8_589_934_592,  # 8GB
            compute_capability=(8, 6),
            temperature=65,
            utilization=80
        )
        ```
    """
    index: int
    name: str
    vendor: GPUVendor
    total_memory: int
    available_memory: int
    compute_capability: Optional[tuple] = None
    temperature: Optional[int] = None
    utilization: Optional[int] = None


class MultiGPUManager:
    """Manager class for handling multiple GPUs across different vendors.
    
    This class provides a unified interface for:
    - GPU detection and initialization
    - Memory management and optimization
    - Model distribution and parallelization
    - Runtime statistics monitoring
    
    Attributes:
        available_gpus (List[GPUInfo]): List of detected and initialized GPUs
        active_gpus (List[GPUInfo]): Currently active and usable GPUs
        vendor_backends (Dict[GPUVendor, Callable]): Vendor-specific setup functions

    Example:
        ```python
        manager = MultiGPUManager()
        
        # Check available GPUs
        if manager.available_gpus:
            # Optimize memory usage
            manager.optimize_memory()
            
            # Get memory statistics
            stats = manager.get_memory_stats()
            print(f"GPU Stats: {stats}")
        ```
    """

    def __init__(self):
        """Initialize the GPU manager and detect available devices."""
        self.available_gpus: List[GPUInfo] = []
        self.active_gpus: List[GPUInfo] = []
        # Map vendors to their setup functions
        self.vendor_backends: Dict[GPUVendor, Callable] = {
            GPUVendor.NVIDIA: self._setup_nvidia,
            GPUVendor.AMD: self._setup_amd,
            GPUVendor.INTEL: self._setup_intel,
        }
        self._detect_gpus()

    def _detect_gpus(self):
        """Detect and initialize all available GPUs across vendors.
        
        This method:
        1. Checks for NVIDIA GPUs using CUDA
        2. Checks for AMD GPUs using ROCm/HIP
        3. Checks for Intel GPUs using OneAPI/IPEX
        4. Falls back to CPU if no GPUs are detected
        
        Note:
            GPU detection order matters for systems with multiple vendor drivers.
            NVIDIA is checked first due to better driver stability and support.
        """
        self.available_gpus = []

        # Primary check for NVIDIA GPUs (most common)
        if torch.cuda.is_available():
            self._setup_nvidia()

        # Secondary check for AMD GPUs
        if hasattr(torch, "hip") and torch.hip.is_available():
            self._setup_amd()

        # Tertiary check for Intel GPUs (Linux only currently)
        if platform.system() == "Linux":
            try:
                import intel_extension_for_pytorch as ipex
                self._setup_intel()
            except ImportError:
                pass

        if not self.available_gpus:
            logger.warning("No GPUs detected. Falling back to CPU.")

    def _setup_nvidia(self):
        """Initialize and configure NVIDIA GPUs.
        
        This method:
        1. Detects all CUDA-capable devices
        2. Retrieves device properties and capabilities
        3. Monitors memory allocation and availability
        4. Sets up temperature and utilization monitoring
        
        Note:
            Some monitoring features may not be available on all NVIDIA GPUs
            or driver versions.
        """
        for i in range(torch.cuda.device_count()):
            # Get device properties and memory info
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            available_memory = total_memory - allocated_memory

            # Create GPU info object with all available metrics
            gpu_info = GPUInfo(
                index=i,
                name=props.name,
                vendor=GPUVendor.NVIDIA,
                total_memory=total_memory,
                available_memory=available_memory,
                compute_capability=(props.major, props.minor),
                temperature=(
                    torch.cuda.get_device_temperature(i)
                    if hasattr(torch.cuda, "get_device_temperature")
                    else None
                ),
                utilization=(
                    torch.cuda.utilization(i)
                    if hasattr(torch.cuda, "utilization")
                    else None
                ),
            )
            self.available_gpus.append(gpu_info)
            logger.info(f"Detected NVIDIA GPU {i}: {props.name}")

    def _setup_amd(self):
        """Initialize and configure AMD GPUs using ROCm/HIP.
        
        This method:
        1. Checks for ROCm/HIP support
        2. Initializes AMD GPU devices
        3. Retrieves basic device properties
        
        Note:
            AMD GPU support requires ROCm installation and may have
            limited functionality compared to NVIDIA GPUs.
        """
        if hasattr(torch, "hip") and torch.hip.is_available():
            try:
                import torch_hip
                for i in range(torch_hip.device_count()):
                    props = torch_hip.get_device_properties(i)
                    gpu_info = GPUInfo(
                        index=i,
                        name=props.name,
                        vendor=GPUVendor.AMD,
                        total_memory=props.total_memory,
                        available_memory=props.total_memory,  # Approximate
                    )
                    self.available_gpus.append(gpu_info)
                    logger.info(f"Detected AMD GPU {i}: {props.name}")
            except Exception as e:
                logger.warning(f"Error setting up AMD GPU: {e}")

    def _setup_intel(self):
        """Initialize and configure Intel GPUs using OneAPI/IPEX.
        
        This method:
        1. Checks for Intel Extension for PyTorch (IPEX)
        2. Initializes Intel GPU devices
        3. Retrieves basic device properties
        
        Note:
            Intel GPU support is currently limited to Linux systems
            and requires IPEX installation.
        """
        try:
            import intel_extension_for_pytorch as ipex
            if ipex.xpu.is_available():
                for i in range(ipex.xpu.device_count()):
                    props = ipex.xpu.get_device_properties(i)
                    gpu_info = GPUInfo(
                        index=i,
                        name=props.name,
                        vendor=GPUVendor.INTEL,
                        total_memory=props.total_memory,
                        available_memory=props.total_memory,  # Approximate
                    )
                    self.available_gpus.append(gpu_info)
                    logger.info(f"Detected Intel GPU {i}: {props.name}")
        except Exception as e:
            logger.warning(f"Error setting up Intel GPU: {e}")

    def get_optimal_devices(self) -> List[torch.device]:
        """Get a prioritized list of optimal devices for computation.
        
        This method returns devices in order of computational efficiency:
        1. NVIDIA GPUs (fastest for deep learning)
        2. AMD GPUs (good for compute workloads)
        3. Intel GPUs (integrated graphics support)
        4. CPU (fallback option)

        Returns:
            List[torch.device]: Ordered list of available compute devices

        Example:
            ```python
            manager = MultiGPUManager()
            devices = manager.get_optimal_devices()
            
            # Use the first (best) device
            model = model.to(devices[0])
            ```
        """
        devices = []

        # Add devices in priority order
        for gpu in self.available_gpus:
            if gpu.vendor == GPUVendor.NVIDIA:
                devices.append(torch.device(f"cuda:{gpu.index}"))
            elif gpu.vendor == GPUVendor.AMD:
                devices.append(torch.device(f"hip:{gpu.index}"))
            elif gpu.vendor == GPUVendor.INTEL:
                devices.append(torch.device(f"xpu:{gpu.index}"))

        # Fallback to CPU if no GPUs available
        if not devices:
            devices.append(torch.device("cpu"))

        return devices

    def distribute_model(self, model, batch_size: int = 1):
        """Distribute a model across available GPUs for parallel processing.
        
        This method implements two parallelization strategies:
        1. Model-specific parallelization (if supported)
        2. DataParallel as a fallback option

        Args:
            model: PyTorch model to distribute
            batch_size (int, optional): Batch size for data parallelism. Defaults to 1.

        Returns:
            Distributed model ready for parallel processing

        Example:
            ```python
            manager = MultiGPUManager()
            model = MyModel()
            
            # Distribute model across available GPUs
            distributed_model = manager.distribute_model(model, batch_size=32)
            ```
        """
        if len(self.available_gpus) <= 1:
            return model

        if hasattr(model, "parallelize"):
            # Use model's built-in parallelization (e.g., for transformers)
            device_map = self._create_device_map()
            return model.parallelize(device_map)
        else:
            # Use DataParallel as fallback
            devices = [gpu.index for gpu in self.available_gpus]
            return torch.nn.DataParallel(model, device_ids=devices)

    def _create_device_map(self) -> Dict[str, int]:
        """Create an optimal mapping of model layers to GPU devices.
        
        This method:
        1. Analyzes available GPU resources
        2. Creates a balanced distribution of model layers
        3. Considers GPU memory and compute capabilities

        Returns:
            Dict[str, int]: Mapping of layer names to GPU indices

        Note:
            Currently optimized for transformer models with 12 layers.
            May need adjustment for different architectures.
        """
        device_map = {}
        gpu_count = len(self.available_gpus)

        if gpu_count <= 1:
            return {"": 0}  # Single GPU or CPU mapping

        # Distribute layers evenly across available GPUs
        for i, gpu in enumerate(self.available_gpus):
            start_idx = i * (12 // gpu_count)
            end_idx = (i + 1) * (12 // gpu_count)

            for layer in range(start_idx, end_idx):
                device_map[f"layer.{layer}"] = gpu.index

        return device_map

    def optimize_memory(self):
        """Optimize GPU memory usage across all available devices.
        
        This method:
        1. Clears unused memory caches
        2. Performs vendor-specific optimizations
        3. Synchronizes devices when necessary

        Example:
            ```python
            manager = MultiGPUManager()
            
            # Run some GPU operations
            # ...
            
            # Optimize memory usage
            manager.optimize_memory()
            ```
        """
        for gpu in self.available_gpus:
            if gpu.vendor == GPUVendor.NVIDIA:
                with torch.cuda.device(gpu.index):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            elif gpu.vendor == GPUVendor.AMD:
                try:
                    import torch_hip
                    torch_hip.empty_cache()
                except ImportError:
                    pass
            elif gpu.vendor == GPUVendor.INTEL:
                try:
                    import intel_extension_for_pytorch as ipex
                    ipex.xpu.empty_cache()
                except ImportError:
                    pass

    def get_memory_stats(self) -> Dict[str, List[Dict]]:
        """Get comprehensive memory statistics for all GPUs.
        
        This method collects:
        - Basic memory information (total, available)
        - Temperature data (if available)
        - Utilization metrics (if available)
        
        Returns:
            Dict[str, List[Dict]]: Nested dictionary of GPU statistics by vendor

        Example:
            ```python
            manager = MultiGPUManager()
            stats = manager.get_memory_stats()
            
            # Print NVIDIA GPU stats
            if stats["nvidia"]:
                for gpu in stats["nvidia"]:
                    print(f"GPU {gpu['index']}: {gpu['utilization']}% used")
            ```
        """
        stats = {vendor.value: [] for vendor in GPUVendor}

        for gpu in self.available_gpus:
            # Collect basic GPU information
            gpu_stats = {
                "index": gpu.index,
                "name": gpu.name,
                "total_memory": gpu.total_memory,
                "available_memory": gpu.available_memory,
            }

            # Add optional metrics if available
            if gpu.temperature is not None:
                gpu_stats["temperature"] = gpu.temperature
            if gpu.utilization is not None:
                gpu_stats["utilization"] = gpu.utilization

            stats[gpu.vendor.value].append(gpu_stats)

        return stats
