import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from typing import Literal
from cs336_basics.nn import Linear, silu

class Positionwise(nn.Module):

    def __init__(self, input_dim: int, 
                 output_dim: int, 
                 dtype=None, 
                 device=None, 
                 activation: Literal["silu", "swiglu"] = "swiglu"):
        
        super(Positionwise, self).__init__()
        self.input_dim = input_dim
        self.d_ff =  4 * self.input_dim if activation == "silu" else int(((8/3) * input_dim) - (((8/3)*input_dim)%64))
        assert self.d_ff % 64 == 0, "d_ff is not multiple of 64"
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        self.activation = activation
        self.w1 = Linear(input_dim, self.d_ff, dtype=self.dtype, device=self.device)
        self.w2 = Linear(self.d_ff, self.output_dim, dtype=self.dtype, device=self.device)
        self.w3 = Linear(input_dim, self.d_ff, dtype=self.dtype, device=self.device)
        
    def forward(self, x: Float[Tensor, "... input_dim"]) -> Float[Tensor, "... output_dim"]:
        z1 = self.w1(x)
        h1 = silu(z1)
        
        if self.activation == "silu":
            return self.w2(h1)
        else:
            z3 = self.w3(x)
            
            return self.w2(h1 * z3)