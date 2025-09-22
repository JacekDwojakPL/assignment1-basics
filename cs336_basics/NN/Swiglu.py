import torch
from jaxtyping import Float

def swiglu(x: torch.Tensor) -> Float[torch.Tensor, "output_dim"]:
    return x * torch.sigmoid(x)