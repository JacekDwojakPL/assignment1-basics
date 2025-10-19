import torch
from torch import Tensor
from jaxtyping import Float

def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    x_normalized = x - torch.logsumexp(x, dim=-1, keepdim=True)  
    exp = torch.exp(x_normalized)
    sum = torch.sum(exp, dim=dim, keepdim=True)

    return exp / sum