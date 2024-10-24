"""
Activation Checkpointing은 모델 학습 중 메모리 사용을 줄이기 위한 기법.
기본적으로 모델의 각 레이어에서 forward pass 시 activation을 저장하는데, 
이는 backward pass 시 필요함. 하지만 모든 activation를 저장하면 메모리 사용량이 급격히 증가할 수 있음.
체크포인팅은 특정 레이어의 활성화만 저장하고, 나머지는 필요 시 재계산함으로써 메모리 사용을 줄이는 방식.
이로 인해 더 큰 모델이나 더 큰 배치 size를 사용할 수 있게 됨.
"""
from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# 비재진입형(checkpoint_impl=NO_REENTRANT) 체크포인팅 래퍼 정의
non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT
)

check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)

def apply_fsdp_checkpointing(model):
    """
    apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )