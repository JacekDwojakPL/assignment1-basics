import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from einops import einsum

class Linear(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 device=None, 
                 dtype=None):
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        self.mean = 0
        self.std = 2 / (input_dim + output_dim)
        self.w = torch.empty((output_dim, input_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(self.w, self.mean, self.std, -3*self.std, 3*self.std)

    def forward(self, x: Float[Tensor, "batch_dim seq_length input_dim"]) -> Float[Tensor, "batch_dim sequence_length output_dim"]:
        return einsum(x.to(self.device), self.w, "batch_dim seq_length input_dim, output_dim input_dim -> batch_dim seq_length output_dim")
    
    def load_state_dict(self, state_dict):
        self.w = state_dict["weights"]