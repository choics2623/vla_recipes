import json
import os
from typing import List, Union

import fire
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM

NUM_SHARDS = {
    "7B": 1,
    "8B": 1,
    "13B": 2,
    "34B": 4,
    "65B": 8,
    "70B": 8,
}

def write_model(model_path, model_size, output_base_path):
    dtype = torch.bfloat16

    params = json.load(open(os.path.join(output_base_path, "params.json"), "r"))
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    llama_version = 3 if params.get("vocab_size") == 128256 else 2

    if "n_kv_heads" in params:
        num_key_value_heads = params["n_kv_heads"] # for GQA / MQA
        num_local_key_value_heads = num_key_value_heads // num_shards
        key_value_dim = dims_per_head * num_key_value_heads
    else:
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim
    

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    loaded = model.state_dict()

    # permute for sliced rotary
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return (
            w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
            .transpose(1,2)
            .reshape(dim1, dim2)
        )
    
    state_dict = [{} for _ in range(num_shards)]

    def insert(name: str, tensor: Union[List, torch.Tensor]):
        for i in range(num_shards):
            state_dict[i][name] = (
                tensor[i].clone() if isinstance(tensor, list) else tensor
            )
    

