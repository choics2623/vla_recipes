import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json

from vla_recipes.model_checkpointing import save_fsdp_model_checkpoint_full, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_peft_checkpoint, save_model_checkpoint
from vla_recipes.policies import fpSixteen, bfSixteen, get_llama_wrapper
from vla_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from vla_recipes.utils.flop_utils import FlopMeasure

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    print("추후 토크나이저 패딩 토큰 좀 확인 해보셈")
    tokenizer.pad_token = 0
    tokenizer.padding_side = "left"
    
@contextlib.contextmanager
def profile(cfg, local_rank=None)