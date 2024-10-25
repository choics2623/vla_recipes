# AnyPrecisionAdamW: 유연한 정밀도의 AdamW 옵티마이저
# 고정밀 가중치 업데이트를 위한 선택적 Kahan 합산을 포함함.
# momentum, variance 및 auxiliary compensation buffer dtypes에 대한 직접적인 제어를 허용함.
# 선택적 Kahan summation은 가중치 업데이트 시 정밀도 감소를 보상하기 위해 사용됨.
# 이를 통해 BFloat16에서의 전체 학습이 가능하며, 많은 case에서 FP32와 동등하거나 더 나은 결과를 제공함.

# Kahan summation은 부동 소수점 연산에서 발생할 수 있는 정밀도 손실을 줄이기 위한 기법으로, 높은 정밀도의 가중치 업데이트를 가능하게 함.
# Kahan 합산을 사용하면 정확하게 1.0을 반환하지만, 일반적인 합산 방식에서는 정밀도 손실로 인해 0에 가까운 값을 반환할 수 있음.
# numbers = [1e16, 1, -1e16]
# def kahan_sum(numbers):
#     sum = 0.0
#     c = 0.0  # 보상 변수
#     for x in numbers:
#         y = x - c
#         t = sum + y
#         c = (t - sum) - y
#         sum = t
#     return sum
import torch
from torch.optim.optimizer import Optimizer

class AnyPrecisionAdamW(Optimizer):
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            use_kahan_summation=False,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            compensation_buffer_dtype=torch.bfloat16,
    ):
        """
        Args:
            params (iterable): 최적화할 파라미터의 반복 가능한 객체 또는 파라미터 그룹을 정의하는 딕셔너리
            lr (float, optional): 학습률 (기본값: 1e-3)
            betas (Tuple[float, float], optional): 그라디언트 및 그 제곱의 이동 평균을 계산하는 데 사용되는 계수 (기본값: (0.9, 0.999))
            eps (float, optional): 분모에 추가되어 수치적 안정성을 향상시키는 항 (기본값: 1e-8)
            weight_decay (float, optional): 가중치 감소 계수 (기본값: 1e-2)

            # Any Precision 전용
            use_kahan_summation = 높은 정밀도의 모델 파라미터 업데이트를 보장하기 위해 보조 버퍼를 생성 (기본값: False)
            momentum_dtype = 모멘텀의 데이터 타입 (기본값: BFloat32)
            variance_dtype = uncentered variance의 데이터 타입 (기본값: BFloat16)
            compensation_buffer_dtype = Kahan 합산 버퍼의 데이터 타입 (기본값: BFloat16)

            # 사용법
            이 옵티마이저는 최적화 상태와 고정밀 업데이트를 위한 Kahan 합산을 구현하며, 모두 사용자가 제어하는 데이터 타입으로 설정됨.
            기본값은 분산은 BF16, 모멘텀은 FP32임.
            이 옵티마이저는 FSDP mixed precision, AMP 또는 full precision에서 실행할 수 있으며, 사용자가 원하는 학습 파이프라인에 따라 다름.

            use_kahan_summation을 False로 설정하고 모멘텀과 분산의 데이터 타입을 FP32로 변경하면, 이는 표준 AdamW 옵티마이저로 되돌아감.
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_kahan_summation=use_kahan_summation,
            momentum_dtype=momentum_dtype,
            variance_dtype=variance_dtype,
            compensation_buffer_dtype=compensation_buffer_dtype,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Args:
            closure(callable, optional): 모델을 재평가하고 손실을 반환하는 closure function임
        """

        if closure is not None:
            with torch.enable_grad():
                # to fix linter, we do not keep the returned loss for use atm
                closure()
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AnyPrecisionAdamW does not support sparse gradients"
                    )
                
                state = self.state[p]

                # State initializatoin
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        dtype=momentum_dtype
                    )

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        dtype=variance_dtype
                    )

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(
                            p,
                            dtype=compensation_buffer_dtype
                        )

            # main processing ---
            # update the steps for each param group update
            state["step"] += 1
            step = state["step"]

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            grad = p.grad

            # weight decay, AdamW style
            if weight_decay:
                p.data.mul_(1 - lr * weight_decay)
            

            # update momentum
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # update uncentered variance
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # adjust using bias1
            bias_correction1 = 1 - beta1**step

            step_size = lr / bias_correction1

            # adjust using bias2
            denom_correction = (1 - beta2**step) ** 0.5 # avoids math import

            centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(
                eps, alpha=1
            )

            # lr update to compensation
            if use_kahan_summation:
                compensation = state["compensation"]

                compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                # update weights with compensation (Kahan summation)
                # save error back to compensation for next iteration
                temp_buffer = p.detach().clone()
                p.data.add_(compensation)

                compensation.add_(temp_buffer.sub_(p.data))

            else:
                # usual AdamW updates
                p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)