import torch


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


if torch.cuda.is_available() or is_xpu_available():
    from .tts_gpu import TTSGPUModel as TTSModel
else:
    from .tts_cpu import TTSCPUModel as TTSModel

__all__ = ["TTSModel"]
