"""Memory management module for multi-vendor GPU and system memory optimization.

This module provides comprehensive memory management capabilities for deep learning
workloads across different GPU vendors (NVIDIA, AMD, Intel) and system memory.
Key features include:

- Multi-vendor GPU memory management (NVIDIA, AMD, Intel)
- System memory monitoring and optimization
- Memory pressure tracking and automatic optimization
- Detailed memory statistics and logging
- Automatic cleanup and garbage collection
- Memory fragmentation management
- Temperature and utilization monitoring

Example:
    Basic usage:
    ```python
    from core.memory_manager import MemoryManager
    
    # Initialize memory manager with 90% threshold
    memory_manager = MemoryManager(memory_threshold=0.90)
    
    # Get current memory usage
    gpu_stats = memory_manager.get_gpu_memory_usage(0)
    system_stats = memory_manager.get_system_memory_info()
    
    # Optimize memory if needed
    memory_manager.optimize_memory_allocation()
    
    # Cleanup when done
    memory_manager.cleanup()
    ```

Note:
    This module automatically handles vendor-specific memory management and
    provides appropriate fallbacks when certain GPU types are unavailable.
"""

from datetime import datetime
import os
import gc
import torch
import psutil
import platform
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from config.logging_config import logger


class GPUVendor(Enum):
    """Enumeration of supported GPU vendors.
    
    Attributes:
        NVIDIA: NVIDIA GPUs (CUDA)
        AMD: AMD GPUs (ROCm/HIP)
        INTEL: Intel GPUs (OneAPI)
        UNKNOWN: Unidentified GPU vendor
    """
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """Data structure for GPU device information.
    
    Attributes:
        index (int): Device index
        name (str): GPU model name
        vendor (GPUVendor): GPU manufacturer
        total_memory (int): Total memory in bytes
        available_memory (int): Available memory in bytes
        compute_capability (Optional[tuple]): CUDA compute capability
        temperature (Optional[int]): Current temperature in Celsius
        utilization (Optional[int]): Current utilization percentage
    """
    index: int
    name: str
    vendor: GPUVendor
    total_memory: int
    available_memory: int
    compute_capability: Optional[tuple] = None
    temperature: Optional[int] = None
    utilization: Optional[int] = None


class MemoryManager:
    """Memory manager for GPU and system memory optimization.
    
    This class provides comprehensive memory management capabilities including:
    - Multi-vendor GPU detection and initialization
    - Memory usage monitoring and optimization
    - Automatic cleanup and garbage collection
    - Temperature and utilization tracking
    - Memory pressure management
    
    Attributes:
        memory_threshold (float): Maximum memory utilization threshold
        available_gpus (List[GPUInfo]): List of available GPU devices
        memory_stats (Dict): Memory usage statistics and history
        device (torch.device): Primary compute device
    
    Example:
        ```python
        manager = MemoryManager(memory_threshold=0.85)
        
        # Monitor memory usage
        gpu_memory = manager.get_gpu_memory_usage(0)
        if gpu_memory['pressure'] > 0.8:
            manager.optimize_memory_allocation()
        
        # Get system memory info
        sys_memory = manager.get_system_memory_info()
        if sys_memory['status'] == 'critical':
            manager.cleanup()
        ```
    """

    def __init__(self, memory_threshold: float = 0.90):
        """Initialize memory manager with specified threshold.
        
        Args:
            memory_threshold (float, optional): Maximum memory utilization threshold
                between 0 and 1. Defaults to 0.90 (90%).
        """
        self.memory_threshold = memory_threshold
        self.available_gpus: List[GPUInfo] = []
        self.memory_stats = self._initialize_memory_stats()
        self._detect_gpus()
        self.device = self._get_optimal_device()
        # Initialize CUDA settings early for optimal performance
        self._initialize_cuda_settings()

    def _initialize_cuda_settings(self):
        """Initialize CUDA settings for optimal memory usage.
        
        Configures:
        - Memory allocation strategy
        - TF32 precision
        - cuDNN benchmarking
        - Memory fractions
        - Peak memory tracking
        """
        if torch.cuda.is_available():
            try:
                # Configure memory allocation strategy
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                    "expandable_segments:True,max_split_size_mb:512"
                )

                # Enable TF32 for better performance on Ampere+ GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # Enable cuDNN autotuner
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True

                # Set conservative memory fraction
                torch.cuda.set_per_process_memory_fraction(0.95)

                # Reset memory tracking
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

                # Set primary device
                torch.cuda.set_device(0)

                logger.info("CUDA settings initialized successfully")

            except Exception as e:
                logger.warning(f"Error initializing CUDA settings: {e}")

    def _initialize_memory_stats(self) -> Dict[str, Any]:
        """Initialize memory statistics tracking structure.
        
        Returns:
            Dict[str, Any]: Initial memory statistics structure
        """
        return {
            "peak_gpu_usage": 0.0,
            "peak_system_memory": 0.0,
            "oom_events": 0,
            "total_generation_time": 0.0,
            "successful_generations": 0,
            "failed_generations": 0,
            "last_device_memory": {},
            "gpu_temperature_history": [],
            "memory_warnings": [],
            "vendor_specific": {
                "nvidia": {},
                "amd": {},
                "intel": {},
            },
            "memory_fragments": [],
            "allocation_history": [],
        }

    def _detect_gpus(self):
        """Detect and initialize all available GPUs from different vendors."""
        self.available_gpus = []

        # Primary check for NVIDIA GPUs
        if torch.cuda.is_available():
            self._detect_nvidia_gpus()

        # Secondary check for AMD GPUs
        if hasattr(torch, "hip") and torch.hip.is_available():
            self._detect_amd_gpus()

        # Tertiary check for Intel GPUs (Linux only)
        if platform.system() == "Linux":
            self._detect_intel_gpus()

        if not self.available_gpus:
            logger.warning("No GPUs detected. Using CPU.")

    def _detect_nvidia_gpus(self):
        """Detect and initialize NVIDIA GPUs using CUDA."""
        try:
            for i in range(torch.cuda.device_count()):
                # Get device properties and memory info
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                available_memory = total_memory - allocated_memory

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
        except Exception as e:
            logger.warning(f"Error detecting NVIDIA GPUs: {e}")

    def _detect_amd_gpus(self):
        """Detect and initialize AMD GPUs using ROCm/HIP."""
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
            logger.debug(f"AMD GPU detection skipped: {e}")

    def _detect_intel_gpus(self):
        """Detect and initialize Intel GPUs using OneAPI/IPEX."""
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
            logger.debug(f"Intel GPU detection skipped: {e}")

    def _get_optimal_device(self) -> torch.device:
        """Select the optimal GPU device based on availability and capabilities.
        
        Returns:
            torch.device: Selected compute device (GPU or CPU)
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            return torch.device("cpu")

        try:
            # Use first GPU for simplicity and stability
            device = torch.device("cuda:0")

            # Configure memory growth
            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            # Log selected device
            props = torch.cuda.get_device_properties(0)
            logger.info(f"Selected GPU 0: {props.name}")

            return device

        except Exception as e:
            logger.error(f"Error selecting GPU: {str(e)}")
            return torch.device("cpu")

    def _get_nvidia_memory_usage(self, device_id: int) -> Dict[str, float]:
        """Get detailed memory usage statistics for NVIDIA GPU.
        
        Args:
            device_id (int): GPU device index
            
        Returns:
            Dict[str, float]: Memory usage statistics in MB and percentages
        """
        try:
            # Get device properties and memory info
            props = torch.cuda.get_device_properties(device_id)
            total = props.total_memory
            reserved = torch.cuda.memory_reserved(device_id)
            allocated = torch.cuda.memory_allocated(device_id)
            free = total - allocated

            # Calculate memory pressure
            usage_percent = (allocated / total) * 100
            self.memory_stats["peak_gpu_usage"] = max(
                self.memory_stats["peak_gpu_usage"], usage_percent
            )

            # Get temperature if available
            try:
                temperature = (
                    torch.cuda.get_device_temperature(device_id)
                    if hasattr(torch.cuda, "get_device_temperature")
                    else None
                )
                if temperature is not None:
                    self.memory_stats["gpu_temperature_history"].append(temperature)
            except:
                temperature = None

            return {
                "total": total / (1024**2),  # Convert to MB
                "reserved": reserved / (1024**2),
                "allocated": allocated / (1024**2),
                "free": free / (1024**2),
                "utilization": (
                    torch.cuda.utilization(device_id)
                    if hasattr(torch.cuda, "utilization")
                    else 0
                ),
                "temperature": temperature,
                "pressure": usage_percent / 100,
            }
        except Exception as e:
            logger.error(f"Error getting NVIDIA GPU memory usage: {str(e)}")
            return {}

    def _get_amd_memory_usage(self, device_id: int) -> Dict[str, float]:
        """Get memory usage statistics for AMD GPU.
        
        Args:
            device_id (int): GPU device index
            
        Returns:
            Dict[str, float]: Memory usage statistics in MB
        """
        try:
            import torch_hip

            props = torch_hip.get_device_properties(device_id)
            total = props.total_memory
            return {
                "total": total / (1024**2),
                "free": total / (1024**2),  # Approximate
                "pressure": 0.0,  # Placeholder
            }
        except Exception as e:
            logger.error(f"Error getting AMD GPU memory usage: {str(e)}")
            return {}

    def _get_intel_memory_usage(self, device_id: int) -> Dict[str, float]:
        """Get memory usage statistics for Intel GPU.
        
        Args:
            device_id (int): GPU device index
            
        Returns:
            Dict[str, float]: Memory usage statistics in MB
        """
        try:
            import intel_extension_for_pytorch as ipex

            props = ipex.xpu.get_device_properties(device_id)
            total = props.total_memory
            return {
                "total": total / (1024**2),
                "free": total / (1024**2),  # Approximate
                "pressure": 0.0,  # Placeholder
            }
        except Exception as e:
            logger.error(f"Error getting Intel GPU memory usage: {str(e)}")
            return {}

    def setup_torch_cuda(self):
        """Configure PyTorch CUDA settings for optimal performance."""
        if not torch.cuda.is_available():
            return

        try:
            # Enable performance optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

            # Set conservative memory fraction
            torch.cuda.set_per_process_memory_fraction(0.95)

            # Log configuration
            logger.info("CUDA Configuration:")
            logger.info(f"  CUDA Version: {torch.version.cuda}")
            logger.info(f"  Memory Fraction: 0.95")
            logger.info(f"  TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")

        except Exception as e:
            logger.warning(f"Error configuring CUDA: {str(e)}")

    def _log_gpu_config(self):
        """Log detailed configuration for all detected GPUs."""
        for gpu in self.available_gpus:
            logger.info(f"\n{gpu.vendor.value.upper()} GPU Configuration:")
            logger.info(f"  Device: {gpu.name}")
            logger.info(f"  Total Memory: {gpu.total_memory / (1024**3):.2f} GB")
            if gpu.compute_capability:
                logger.info(
                    f"  Compute Capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
                )
            if gpu.temperature is not None:
                logger.info(f"  Temperature: {gpu.temperature}°C")
            if gpu.utilization is not None:
                logger.info(f"  Utilization: {gpu.utilization}%")

    def optimize_memory_allocation(self):
        """Optimize memory allocation across all available GPUs.
        
        This method:
        1. Forces garbage collection
        2. Clears CUDA cache
        3. Synchronizes devices
        4. Manages memory fragmentation
        5. Records optimization history
        """
        if not torch.cuda.is_available():
            return

        try:
            # Initial cleanup
            gc.collect()

            # Clear CUDA cache with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    continue

            # Optimize each device
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    mem_info = self.get_gpu_memory_usage(device_id)

                    if (mem_info.get("pressure", 0)) > self.memory_threshold:
                        # Record current state
                        self.memory_stats["last_device_memory"][device_id] = mem_info

                        # Enhanced cleanup sequence
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                        # Reset memory tracking
                        torch.cuda.reset_peak_memory_stats()
                        if hasattr(torch.cuda, "memory_stats"):
                            torch.cuda.memory_stats_as_nested_dict()
                            torch.cuda.reset_accumulated_memory_stats()

                        # Configure memory allocation strategy
                        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                                "expandable_segments:True,max_split_size_mb:512"
                            )

                        # Record optimization results
                        self.memory_stats["allocation_history"].append(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "device": device_id,
                                "action": "optimization",
                                "pressure_before": mem_info.get("pressure", 0),
                                "pressure_after": self.get_gpu_memory_usage(
                                    device_id
                                ).get("pressure", 0),
                            }
                        )

        except Exception as e:
            logger.warning(f"Error during memory optimization: {str(e)}")
            self.memory_stats["oom_events"] += 1

    def get_gpu_memory_usage(
        self, device_id: int, vendor: GPUVendor = GPUVendor.NVIDIA
    ) -> Dict[str, float]:
        """Get memory usage statistics for any GPU vendor.
        
        Args:
            device_id (int): GPU device index
            vendor (GPUVendor, optional): GPU vendor. Defaults to NVIDIA.
            
        Returns:
            Dict[str, float]: Memory usage statistics
        """
        try:
            if vendor == GPUVendor.NVIDIA and torch.cuda.is_available():
                return self._get_nvidia_memory_usage(device_id)
            elif vendor == GPUVendor.AMD and hasattr(torch, "hip"):
                return self._get_amd_memory_usage(device_id)
            elif vendor == GPUVendor.INTEL:
                return self._get_intel_memory_usage(device_id)
            return {}
        except Exception as e:
            logger.error(f"Error getting GPU memory usage: {str(e)}")
            return {}

    def get_system_memory_info(self) -> Dict[str, Dict[str, float]]:
        """Get detailed system memory information.
        
        Returns:
            Dict containing:
            - RAM usage statistics
            - Swap memory statistics
            - Memory pressure indicators
            - System status
        """
        try:
            # Get memory information
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Calculate memory pressure
            memory_pressure = {
                "ram_pressure": mem.percent / 100,
                "swap_pressure": swap.percent / 100 if swap.total > 0 else 0,
                "high_pressure": mem.percent > 85
                or (swap.total > 0 and swap.percent > 80),
            }

            # Compile memory statistics
            info = {
                "ram": {
                    "total": mem.total / (1024**3),  # Convert to GB
                    "available": mem.available / (1024**3),
                    "used": mem.used / (1024**3),
                    "percent": mem.percent,
                    "pressure": memory_pressure["ram_pressure"],
                },
                "swap": {
                    "total": swap.total / (1024**3),
                    "used": swap.used / (1024**3),
                    "percent": swap.percent,
                    "pressure": memory_pressure["swap_pressure"],
                },
                "status": "critical" if memory_pressure["high_pressure"] else "normal",
            }

            # Update peak memory usage
            self.memory_stats["peak_system_memory"] = max(
                self.memory_stats["peak_system_memory"], memory_pressure["ram_pressure"]
            )

            return info
        except Exception as e:
            logger.error(f"Error getting system memory info: {str(e)}")
            return {"ram": {}, "swap": {}, "status": "unknown"}

    def optimize_memory_allocation(self):
        """Optimize memory allocation and cleanup"""
        if not torch.cuda.is_available():
            return

        gc.collect()
        torch.cuda.empty_cache()

        try:
            for device_id in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device_id)
                mem_info = self.get_gpu_memory_usage(device_id)

                if (mem_info.get("pressure", 0)) > self.memory_threshold:
                    self.memory_stats["last_device_memory"][device_id] = mem_info
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device_id)

                    if hasattr(torch.cuda, "memory_stats"):
                        torch.cuda.memory_stats_as_nested_dict()
                        torch.cuda.reset_accumulated_memory_stats()

        except Exception as e:
            logger.warning(f"Error during memory optimization: {str(e)}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics for all GPUs.
        
        Returns:
            Dict containing:
            - Peak memory usage
            - Temperature history
            - Memory warnings
            - Vendor-specific statistics
            - Allocation history
        """
        stats = self.memory_stats.copy()

        # Add current GPU states
        for gpu in self.available_gpus:
            vendor_stats = stats["vendor_specific"][gpu.vendor.value]
            vendor_stats[f"gpu_{gpu.index}"] = self.get_gpu_memory_usage(
                gpu.index, gpu.vendor
            )

        return stats

    def cleanup(self):
        """Perform comprehensive memory cleanup across all devices.
        
        This method:
        1. Forces garbage collection
        2. Clears GPU memory caches
        3. Synchronizes devices
        4. Resets memory statistics
        5. Records cleanup history
        """
        try:
            # Force garbage collection
            gc.collect()

            if torch.cuda.is_available():
                # Synchronize before cleanup
                torch.cuda.synchronize()

                # Clear CUDA cache
                torch.cuda.empty_cache()

                # Reset memory statistics
                if hasattr(torch.cuda, "memory_stats"):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()

                # Clear GPU tensors
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj):
                            if obj.is_cuda:
                                del obj
                    except Exception:
                        pass

                # Vendor-specific cleanup
                for gpu in self.available_gpus:
                    if gpu.vendor == GPUVendor.NVIDIA:
                        with torch.cuda.device(gpu.index):
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    elif gpu.vendor == GPUVendor.AMD and hasattr(torch, "hip"):
                        torch.hip.empty_cache()
                    elif gpu.vendor == GPUVendor.INTEL:
                        try:
                            import intel_extension_for_pytorch as ipex
                            ipex.xpu.empty_cache()
                        except ImportError:
                            pass

            # Reset memory tracking
            self.memory_stats["last_device_memory"].clear()

            # Record successful cleanup
            self.memory_stats["allocation_history"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "cleanup",
                    "success": True,
                }
            )

        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
            self.memory_stats["allocation_history"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "cleanup",
                    "success": False,
                    "error": str(e),
                }
            )

        finally:
            # Final garbage collection
            gc.collect()
