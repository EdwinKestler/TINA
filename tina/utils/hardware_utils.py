import torch

def detect_hardware():
    """
    Detects available hardware for PyTorch execution (NVIDIA, AMD, or CPU).
    Returns 'nvidia', 'amd', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return "nvidia"
    elif torch.backends.mps.is_available():
        return "mps"
    elif torch.has_mps:
        return "amd"
    else:
        return "cpu"
