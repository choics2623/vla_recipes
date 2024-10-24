from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class fsdp_config:
    mixed_precision: bool=True
    use_fp16: bool=False
    # Model Parameter과 Optimizer 상태를 노드 내부에서 풀 샤딩(분할)하고, 노드간에서는 Distributed Data Parallel(DDP)를 사용하는 전략
    # HYBRID_SHARD: 노드 내에서 풀 샤드, 노드간에는 DDP 사용
    # SHARD_GRAD_OP: Gradients와 Optimizer 상태만 샤딩
    # NO_SHARD: 샤딩하지 않고 DDP와 비슷하게 사용
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    # HYBRID_SHARD전략을 사용할때만 사용 가능 - 커스터마이징된 GPU 수(Sharding_group)로 모델을 샤딩하고 Sharding_group위에 Replica를 추가하여 사용 -> 좀 더 자원 효율적으로 사용하기 위한 옵션 인듯
    hsdp : bool =False
    # HYBRID_SHARD전략을 사용할때만 사용 가능 - 모델이 샤딩 될 수 있는 GPU의 수를 의미 함 -> 모델의 replicas가 들어 갈 수 있는 GPU의 수를 정의함
    sharding_group_size : int=0
    # HYBRID_SHARD전략을 사용할때만 사용 가능 - 모델 replica의 크기를 결정함. 
    # 예를 들어, world_size가 16이고, sharding_group_size가 4라면, replica_group_size는 16 / 4 = 4가 됩니다. 이는 4개의 GPU 그룹이 모델의 하나의 replica를 담당한다는 의미
    replica_group_size: int=0
    # 체크포인트 저장 방식을 결정하는 옵션
    # SHARDED_STATE_DICT: 랭크별로 나뉜 가중치가 각각 파일로 저장됨
    # FULL_STATE_DICT: 모든 가중치를 랭크 0에서 모아 하나의 파일로 저장함
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    # True로 설정하면, 모델의 중간 Activation을 저장하여 메모리 사용량을 줄이고, 메모리 오프로드를 효과적으로 수행할 수 있음
    fsdp_activation_checkpointing: bool=True
    # 모델의 파라미터와 Optimizer state를 CPU로 오프로드 할지 여부를 결정함.
    # True로 설정하면, GPU 메모리가 부족할때 CPU로 오프로드하여 GPU 메모리 사용량을 줄일 수 있음
    fsdp_cpu_offload: bool=False
    pure_bf16: bool=False
    optimizer: str= "AdamW"
