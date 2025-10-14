import torch
from torch import Tensor
from jaxtyping import Float
from einops import rearrange

def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    max_element = torch.max(x, dim=dim)
    x_normalized = x - rearrange(max_element.values, "... (b e) -> ... b e", e=1)
    exp = torch.exp(x_normalized)
    sum = rearrange(torch.sum(exp, dim=dim), "... (b e) -> ... b e", e=1)

    return exp / sum