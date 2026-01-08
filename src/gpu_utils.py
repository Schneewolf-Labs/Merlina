"""
GPU detection and monitoring utilities.

Provides functions to:
- List available GPUs
- Monitor GPU memory and utilization
- Select specific GPUs for training
"""

import os
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU features disabled")

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("pynvml not available - detailed GPU monitoring disabled")


class GPUInfo:
    """Information about a single GPU."""

    def __init__(
        self,
        index: int,
        name: str,
        total_memory: int,
        free_memory: int,
        used_memory: int,
        utilization: Optional[int] = None,
        temperature: Optional[int] = None,
        power_usage: Optional[int] = None,
        compute_capability: Optional[tuple] = None
    ):
        self.index = index
        self.name = name
        self.total_memory = total_memory  # bytes
        self.free_memory = free_memory    # bytes
        self.used_memory = used_memory    # bytes
        self.utilization = utilization    # percentage
        self.temperature = temperature    # celsius
        self.power_usage = power_usage    # watts
        self.compute_capability = compute_capability

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "index": self.index,
            "name": self.name,
            "total_memory_mb": self.total_memory // (1024 * 1024),
            "free_memory_mb": self.free_memory // (1024 * 1024),
            "used_memory_mb": self.used_memory // (1024 * 1024),
            "memory_utilization_percent": round(
                (self.used_memory / self.total_memory * 100) if self.total_memory > 0 else 0, 1
            ),
            "gpu_utilization_percent": self.utilization,
            "temperature_c": self.temperature,
            "power_usage_w": self.power_usage,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}"
                                 if self.compute_capability else None
        }


class GPUManager:
    """Manages GPU detection and monitoring."""

    def __init__(self):
        self.nvml_initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("NVML initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")

    def __del__(self):
        """Cleanup NVML on deletion."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                # Log cleanup errors at debug level (don't raise in destructor)
                # Using print since logger may not be available during shutdown
                import sys
                print(f"[DEBUG] NVML shutdown error (safe to ignore): {e}", file=sys.stderr)

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return TORCH_AVAILABLE and torch.cuda.is_available()

    def get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        if not self.is_cuda_available():
            return 0
        return torch.cuda.device_count()

    def get_gpu_info(self, index: int) -> Optional[GPUInfo]:
        """Get detailed information about a specific GPU."""
        if not self.is_cuda_available():
            return None

        if index >= self.get_gpu_count():
            return None

        try:
            # Basic info from PyTorch
            props = torch.cuda.get_device_properties(index)
            name = props.name
            total_memory = props.total_memory
            compute_capability = (props.major, props.minor)

            # Memory info from PyTorch
            free_memory = total_memory - torch.cuda.memory_allocated(index)
            used_memory = torch.cuda.memory_allocated(index)

            # Additional info from NVML if available
            utilization = None
            temperature = None
            power_usage = None

            if self.nvml_initialized:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)

                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu

                    # Temperature
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )

                    # Power usage
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW to W

                    # Get more accurate memory info from NVML
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory = mem_info.total
                    free_memory = mem_info.free
                    used_memory = mem_info.used

                except Exception as e:
                    logger.debug(f"Could not get NVML stats for GPU {index}: {e}")

            return GPUInfo(
                index=index,
                name=name,
                total_memory=total_memory,
                free_memory=free_memory,
                used_memory=used_memory,
                utilization=utilization,
                temperature=temperature,
                power_usage=power_usage,
                compute_capability=compute_capability
            )

        except Exception as e:
            logger.error(f"Error getting GPU {index} info: {e}")
            return None

    def list_gpus(self) -> List[GPUInfo]:
        """List all available GPUs with their information."""
        gpus = []
        for i in range(self.get_gpu_count()):
            info = self.get_gpu_info(i)
            if info:
                gpus.append(info)
        return gpus

    def get_available_gpus(self, min_free_memory_mb: int = 0) -> List[int]:
        """
        Get list of GPU indices with at least min_free_memory_mb available.

        Args:
            min_free_memory_mb: Minimum free memory in MB

        Returns:
            List of GPU indices
        """
        available = []
        for gpu in self.list_gpus():
            free_mb = gpu.free_memory // (1024 * 1024)
            if free_mb >= min_free_memory_mb:
                available.append(gpu.index)
        return available

    def set_visible_devices(self, device_ids: List[int]) -> None:
        """
        Set CUDA_VISIBLE_DEVICES environment variable.

        Args:
            device_ids: List of GPU indices to make visible
        """
        if device_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
            logger.info(f"Set CUDA_VISIBLE_DEVICES to: {device_ids}")
        else:
            # Empty list means CPU only
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            logger.info("Set CUDA_VISIBLE_DEVICES to empty (CPU only)")

    def get_current_visible_devices(self) -> Optional[str]:
        """Get current CUDA_VISIBLE_DEVICES setting."""
        return os.environ.get('CUDA_VISIBLE_DEVICES')

    def get_recommended_gpu(self) -> Optional[int]:
        """
        Get the GPU with the most free memory.

        Returns:
            GPU index or None if no GPUs available
        """
        gpus = self.list_gpus()
        if not gpus:
            return None

        # Sort by free memory (descending)
        gpus.sort(key=lambda g: g.free_memory, reverse=True)
        return gpus[0].index


# Global instance
_gpu_manager = None


def get_gpu_manager() -> GPUManager:
    """Get or create the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager
