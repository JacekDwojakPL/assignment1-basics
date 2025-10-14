import torch
from typing import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    norm = sum([torch.sum(p.grad**2) for p in filter(lambda p: p.grad != None, parameters)])
    norm = norm ** 0.5
    
    if norm > max_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad *= (max_norm / (norm+1e-6))