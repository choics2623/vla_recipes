from dataclasses import dataclass, field
from typing import List

@dataclass
class lora_config:
    r: int=8
    lora_alpha: int=32
    target_modules: List[str] = field(default_factory=lambda: ["v_proj","o_proj","q_proj","down_proj","gate_proj","k_proj","up_proj"])
    bias= "none"
    task_type: str= "CAUSAL_LM"
    lora_dropout: float=0.05
    inference_mode: bool=False

@dataclass
class llama_adapter_config:
    adapter_len: int=10
    adapter_layers: int=30
    task_type: str="CAUSAL_LM"

# CAUTION prefix tuning is currently not supported
@dataclass
class prefix_config:
    num_virtual_tokens: int=30
    task_type: str="CAUSAL_LM"