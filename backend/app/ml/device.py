try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from app.core.config import settings
from app.core.logging import logger

def get_device():
    """
    Selects the optimal execution device based on hardware availability and configuration.
    Priority: User config -> CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if not HAS_TORCH:
        return "cpu"

    configured = settings.DEVICE.lower()
    
    if configured == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif configured == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif configured == "cpu":
        return torch.device("cpu")

    # Auto-detection
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using NVIDIA GPU Accelerator: {device_name}")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS Accelerator")
        return torch.device("mps")
    else:
        logger.info("Hardware accelerator not detected or CPU specified. Using CPU.")
        return torch.device("cpu")

CURRENT_DEVICE = get_device()
