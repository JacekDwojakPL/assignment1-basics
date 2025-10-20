import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.zeros(d_model, device=self.device))
        nn.init.normal_(self.weight)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        input_type = x.dtype
        x = x.to(torch.float32)
        mean = self._mean(x)
        result = (x / mean)*self.weight

        return result.to(input_type)

    def _mean(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... 1"]:

        return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
