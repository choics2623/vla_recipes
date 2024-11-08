from torch.distributed.tensor.device_mesh import init_device_mesh
import os

def fsdp_auto_wrap_policy(model, transformer_layer_names):
    import functools

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    def lambda_policy_fn(module):
        if(
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False
    
    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(transformer_layer_names)
    )

    auto_warp_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_warp_policy

# 
def hsdp_device_mesh(replica_group_size, sharding_group_size, device=None):
    """
    이 함수는 FSDP(Fully Sharded Data Parallel) 학습에서 Hybrid Sharding 전략을 사용하기 위해 device mesh를 초기화 하며,
    GPU에 대한 모델 적합성을 알 수 없는 경우를 대비해, 복제(replica)와 샤딩(sharding) 그룹의 크기를 명시적으로 지정할 수 있어 분산 학습 설정에 유연성을 제공 함.
    
    device_mesh: 여러 디바이스(GPU, TPU 등)를 묶어 네트워크 상에서 어떻게 데이터를 분산할지를 정의한 구조

    매개변수:
        replica_group_size (int): 각 복제 그룹의 크기. 사용 가능한 리소스 내에서 모델이 실행될 수 있도록 제공되어야 함.
        sharding_group_size (int): 모델이 맞출 수 있는 각 샤딩 그룹의 크기. 모델 파라미터의 올바른 분배를 보장하기 위해 제공되어야 함.
        device (str, 선택사항): 사용할 디바이스 (예: "cuda:0"). None으로 설정되면, 기본값으로 "cuda"가 사용되고, 로컬 랭크(local rank)가 디바이스 인덱스가 됨.
    Returns:
        FSDP와 호환되는 디바이스 메쉬 객체를 반환.
    Raises:
        ValueError: replica_group_size나 sharding_group_size가 제공되지 않았거나, 월드 사이즈(world size)가 샤딩 그룹 크기로 나누어 떨어지지 않을 때 발생 함.
        RuntimeError: 유효한 디바이스 메쉬를 생성할 수 없을 때 발생 함.

    Usage:
        모델이 4개의 GPU에 맞도록 설정되었고, 3개의 노드에 각각 8개의 GPU가 있는 경우:
        Sharding_Group_Size = 4
        Replica_Groups_Size = (총 24개의 GPU 중 4개가 샤딩 그룹에 할당) = 6개의 복제 그룹
    >>> device_mesh = initialize_device_mesh(replica_group_size, sharding_group_size)
    >>> sharded_model = FSDP(model, device_mesh=device_mesh, ...)
    """

    if replica_group_size is None or sharding_group_size is None:
        raise ValueError("Both replica_group_size and sharding_group_size must be provided")
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    device = device or f"cuda"

    if world_size % sharding_group_size != 0:
        raise ValueError(f"World size {world_size} is not evenly divisible by "
                         f"sharding group size {sharding_group_size}.")
    
    if (world_size // sharding_group_size) % replica_group_size != 0:
        raise ValueError(f"The calculated number of replica groups is not evenly divisible by "
                         f"replica_group_size {replica_group_size}.")
    
    device_mesh = init_device_mesh(device, (replica_group_size, sharding_group_size))
    if device_mesh is None:
        raise RuntimeError("Failed to create a valid device mesh.")
    
    return device_mesh