from typing import Any, Dict, List, Optional, Union
import time
import torch
from torch.utils.flop_counter import FlopCounterMode

class FlopMeasure(FlopCounterMode):
    """
    ``FlopMeasure``은 컨텍스트 내에서 FLOPs(부동소수점 연산)의 수를 계산하는 맞춤형 컨텍스트 매니저임. `FlopCounterMode`를 상속 받아 사용하고,
    추가적인 start_counting() 및 stop_counting() 기능을 포함하여 warming up 단계 이후에만 FLOPs 계산이 시작되도록 함.

    또한 생성 시 FlopCounterMode에 모듈(또는 모듈 리스트)를 전달하여 계층적 출력(hierarchical output)을 지원함.

    사용 예시
    .. code-block:: python

    model = ...
    flop_counter = FlopMeasure(model,local_rank=0,warmup_step=3)
    for batch in enumerate(dataloader):
        with flop_counter:
            model(batch)
            flop_counter.step()
    """

    def __init__(
            self,
            mods: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
            depth: int = 2,
            display: bool = True,
            custom_mapping: Dict[Any, Any] = None,
            rank=None,
            warming_step: int = 3
    ):
        super().__init__(mods, depth, display, custom_mapping)
        self.rank = rank
        self.warmup_step = warming_step
        self.start_time = 0
        self.end_time = 0

    def step(self):
        # decrease the warmup step by 1 for every step, so that the flop counting will start when warmup_step
        if self.warmup_step >= 0:
            self.warmup_step -= 1
        if self.warmup_step == 0 and self.start_time == 0:
            self.start_time = time.time()
        elif self.warmup_step == -1 and self.start_time != 0 and self.end_time == 0:
            self.end_time = time.time()

    def __enter__(self):
        if self.warmup_step == 0:
            self.start_time = time.time()
        super().__enter__()
        return self
    
    def is_done(self):
        return self.warmup_step == -1
    
    def get_total_flops(self) -> int:
        return super().get_total_flops()
    
    def get_flops_per_sec(self):
        if self.start_time == 0 or self.end_time == 0:
            print("Warning: flop count did not finish correctly")
            return 0
        return super().get_total_flops() / (self.end_time - self.start_time)
    
    def get_table(self, depth=2):
        return super().get_table(depth)
    

    def __exit__(self, *args):
        if self.get_total_flops() == 0:
            print(
                "Warning: did not record any flops this time. Skipping the flop report"
            )
        else:
            if self.display:
                if self.rank is None or self.rank == 0:
                    print(f"Total time used in this flop counting step is : {self.end_time - self.start_time}")
                    print(f"The total TFlop per second is: {self.get_flop_counts() / 1e12}")
                    print(f"The tflop_count table is below:")
                    print(self.get_table(self.depth))
            # Disable the display feature so that we don't print the table again
            self.display = False
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        # when warmup_step is 0, count the flops and return the original output
        if self.warmup_step == 0:
            return super().__torch_dispatch__(func, types, args, kwargs)
        # otherwise, just return the original output
        return func(*args, **kwargs)
