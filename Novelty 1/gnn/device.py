"""
CUDA Device Selection and GPU Tuning for the Multi-Task GNN Arm.

Every device decision in this package routes through here, so there is exactly one
place that determines what runs on the GPU.
"""

import os
from typing import Dict, Any

import torch


def get_device(prefer_cuda: bool = True, device_index: int = 0) -> torch.device:
    """
    Returns the compute device, preferring CUDA when available.

    Set the environment variable BBB_GNN_FORCE_CPU=1 to force CPU execution. The
    gradient-masking verification checks use this because they need bitwise
    determinism that TF32 matmuls do not provide.
    """
    if os.environ.get('BBB_GNN_FORCE_CPU', '') == '1':
        return torch.device('cpu')
    if prefer_cuda and torch.cuda.is_available():
        return torch.device(f'cuda:{device_index}')
    return torch.device('cpu')


def describe_device(device: torch.device) -> Dict[str, Any]:
    """Collects a full record of the compute environment for the run log."""
    info: Dict[str, Any] = {
        'device': str(device),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if device.type == 'cuda':
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        info.update({
            'gpu_name': props.name,
            'gpu_total_memory_gb': round(props.total_memory / (1024 ** 3), 2),
            'gpu_capability': f'{props.major}.{props.minor}',
            'gpu_multiprocessors': props.multi_processor_count,
            'cuda_runtime': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
        })
    return info


def tune_backend(device: torch.device, allow_tf32: bool = True) -> Dict[str, Any]:
    """
    Enables the GPU fast paths that are safe for this workload.

    cudnn.benchmark stays ON: molecular graphs vary in size, so autotuning costs a
    little on the first batches and pays back over 200 epochs. TF32 matmul is
    enabled because these are 256-dim dense layers where the precision loss is
    immaterial next to the label noise in BBB data.
    """
    settings: Dict[str, Any] = {'tf32_enabled': False, 'cudnn_benchmark': False}
    if device.type != 'cuda':
        return settings

    torch.backends.cudnn.benchmark = True
    settings['cudnn_benchmark'] = True

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        settings['tf32_enabled'] = True

    return settings


def cuda_memory_summary(device: torch.device) -> Dict[str, float]:
    """Peak GPU memory for the run log. Returns an empty dict on CPU."""
    if device.type != 'cuda':
        return {}
    return {
        'peak_allocated_gb': round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3),
        'peak_reserved_gb': round(torch.cuda.max_memory_reserved(device) / (1024 ** 3), 3),
    }


def reset_cuda_peak(device: torch.device) -> None:
    """
    Resets peak-memory counters so per-run numbers are independent.

    The device object is passed rather than a bare integer index: the integer form
    raises "Invalid device argument" before the CUDA context has been initialised.
    """
    if device.type != 'cuda':
        return
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)


def empty_cuda_cache(device: torch.device) -> None:
    """Releases cached blocks between folds to keep peak usage honest."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
