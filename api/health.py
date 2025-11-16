"""Health check endpoints for monitoring and deployment.

This module provides health check endpoints that can be used by:
- Kubernetes liveness and readiness probes
- Load balancers
- Monitoring systems
- Docker healthchecks

Examples:
    FastAPI integration:
        >>> from fastapi import FastAPI
        >>> from api.health import router
        >>> app = FastAPI()
        >>> app.include_router(router)

    Gradio custom route:
        >>> import gradio as gr
        >>> from api.health import health_check_simple
        >>> demo = gr.Interface(...)
        >>> demo.launch(health_check=health_check_simple)
"""

from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from _version import __version__
except ImportError:
    __version__ = "unknown"

from config import logger


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information for health checks.

    Returns:
        Dict containing GPU status and details
    """
    if not TORCH_AVAILABLE:
        return {
            "available": False,
            "reason": "PyTorch not installed"
        }

    try:
        cuda_available = torch.cuda.is_available()

        if not cuda_available:
            return {
                "available": False,
                "reason": "CUDA not available"
            }

        device_count = torch.cuda.device_count()
        devices = []

        for i in range(device_count):
            try:
                device_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB

                devices.append({
                    "id": i,
                    "name": device_name,
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_reserved_gb": round(memory_reserved, 2),
                    "memory_total_gb": round(memory_total, 2),
                    "memory_free_gb": round(memory_total - memory_reserved, 2)
                })
            except Exception as e:
                logger.warning(f"Failed to get info for GPU {i}: {e}")
                devices.append({
                    "id": i,
                    "error": str(e)
                })

        return {
            "available": True,
            "device_count": device_count,
            "devices": devices
        }

    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        return {
            "available": False,
            "error": str(e)
        }


def health_check_simple() -> Dict[str, str]:
    """Simple health check endpoint.

    Returns basic status without any heavy operations.
    Suitable for frequent polling by load balancers.

    Returns:
        Dict with status field
    """
    return {"status": "healthy"}


def health_check_detailed() -> Dict[str, Any]:
    """Detailed health check with system information.

    Includes GPU status, memory info, and version.
    Use for diagnostic purposes, not frequent polling.

    Returns:
        Dict with comprehensive system information
    """
    gpu_info = get_gpu_info()

    response = {
        "status": "healthy",
        "version": __version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_available": TORCH_AVAILABLE,
    }

    if TORCH_AVAILABLE:
        response["torch_version"] = torch.__version__
        response["gpu"] = gpu_info

    return response


def readiness_check(pipeline=None) -> Dict[str, Any]:
    """Readiness check for Kubernetes.

    Checks if the service is ready to accept requests.
    Should return success (200) only when model is loaded and ready.

    Args:
        pipeline: Optional FluxPipeline instance to check if model is loaded

    Returns:
        Dict with ready status and details
    """
    checks = {
        "torch_available": TORCH_AVAILABLE,
        "gpu_available": False,
        "model_loaded": False
    }

    if TORCH_AVAILABLE:
        gpu_info = get_gpu_info()
        checks["gpu_available"] = gpu_info.get("available", False)

    # Check if pipeline is loaded (if provided)
    if pipeline is not None:
        try:
            checks["model_loaded"] = hasattr(pipeline, 'pipe') and pipeline.pipe is not None
        except Exception as e:
            logger.warning(f"Failed to check pipeline status: {e}")
            checks["model_loaded"] = False

    # Service is ready if torch and GPU are available
    # Model loaded check is optional (may start without preloading)
    ready = checks["torch_available"] and checks["gpu_available"]

    return {
        "ready": ready,
        "checks": checks,
        "message": "Ready to accept requests" if ready else "Not ready"
    }


def liveness_check() -> Dict[str, str]:
    """Liveness check for Kubernetes.

    Simple check to verify the process is alive and responsive.
    Should never perform heavy operations.

    Returns:
        Dict with alive status
    """
    return {"status": "alive"}


# FastAPI router (optional, requires fastapi installed)
try:
    from fastapi import APIRouter, Response, status

    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("")
    async def health_endpoint() -> Dict[str, str]:
        """Basic health check endpoint."""
        return health_check_simple()

    @router.get("/detailed")
    async def detailed_health_endpoint() -> Dict[str, Any]:
        """Detailed health check with system info."""
        return health_check_detailed()

    @router.get("/ready")
    async def readiness_endpoint(response: Response) -> Dict[str, Any]:
        """Kubernetes readiness probe."""
        result = readiness_check()
        if not result["ready"]:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    @router.get("/live")
    async def liveness_endpoint() -> Dict[str, str]:
        """Kubernetes liveness probe."""
        return liveness_check()

except ImportError:
    # FastAPI not installed, skip router creation
    router = None


if __name__ == "__main__":
    # Test health checks
    print("Simple Health Check:")
    print(health_check_simple())
    print("\nDetailed Health Check:")
    print(health_check_detailed())
    print("\nReadiness Check:")
    print(readiness_check())
    print("\nLiveness Check:")
    print(liveness_check())
