import torch
from typing import Iterable, Tuple

class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-3,
                 weight_decay: float = 0.01,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        defaults = {"lr": lr,
                    "initial_lr": lr,
                    "weight_decay": weight_decay, 
                    "betas": betas,
                    "eps": eps}
        super().__init__(params, defaults)


    def step(self, closure=None) -> torch.Tensor | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
        
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                m = (beta1*m) + (1-beta1)*grad
                v = (beta2*v) + (1-beta2)*(grad**2)
                lr_t = lr * (((1-beta2**t)**0.5) / (1-beta1**t))
                p.data -= lr_t * (m / (v**0.5+eps))
                p.data -= lr*weight_decay*p.data 
                state["t"] = t+1
                state["m"] = m
                state["v"] = v

        return loss