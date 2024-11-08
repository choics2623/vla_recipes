import gc
import psutil
import threading

import torch
from accelerate.utils import is_xpu_available

def byte2gb(x):
    return int(x / 2**30)

# This context managet is used to track the peak memory usage of the process
class MemoryTrace:
def __enter__(self):
    gc.collect()
    if is_xpu_available():
        torch.xpu.empty_cache()
        torch.xpu.reset