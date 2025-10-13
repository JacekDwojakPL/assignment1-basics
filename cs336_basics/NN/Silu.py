import torch
from jaxtyping import Float

def silu(x: torch.Tensor) -> Float[torch.Tensor, "output_dim"]:
    return x * torch.sigmoid(x)