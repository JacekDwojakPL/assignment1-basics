from math import cos, pi
from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_iters: int, cosine_cycle_iters: int, lr_min: float = 0, last_epoch: int = -1):
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        self.lr_min = lr_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        timestep = self.last_epoch       
        if timestep < self.warmup_iters:
            warmup_factor = timestep / self.warmup_iters
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        elif timestep <= self.cosine_cycle_iters:
            progress = (timestep - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters)
            cosine_factor = 0.5 * (1 + cos(progress * pi))
            return [self.lr_min + cosine_factor * (base_lr - self.lr_min) 
                    for base_lr in self.base_lrs]
        
        else:
            return [self.lr_min for _ in self.base_lrs]