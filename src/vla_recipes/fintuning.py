from collections import Counter
import os

import dataclasses
import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, PeftModel
from torch.distributed.fsdp import(
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoProcessor,
    LlamaForCausalLM,
    MllamaForConditionalGeneration,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mllama.modeling_mllama import MllamaSelfAttentionDecoderLayer,MllamaCrossAttentionDecoderLayer,MllamaVisionEncoderLayer

from vla_recipes.configs