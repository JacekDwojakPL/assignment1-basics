import torch
import torch.nn as nn
from jaxtyping import Float
from cs336_basics.NN import Linear, swiglu

class Positionwise(nn.Module):
    
    def __init__(self, input_dim, output_dim, dtype=None, device=None):
        super(Positionwise, self).__init__()
        self.input_dim = input_dim
        self.d_ff =  int(((8/3) * input_dim) - (((8/3)*input_dim)%64))
        assert self.d_ff % 64 == 0, "d_ff is not multiple of 64"
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(input_dim, self.d_ff, dtype=self.dtype, device=self.device)
        self.w2 = Linear(self.d_ff, self.output_dim, dtype=self.dtype, device=self.device)
        self.w3 = Linear(input_dim, self.d_ff, dtype=self.dtype, device=self.device)
        
    def forward(self, x: torch.Tensor) -> Float[torch.Tensor, "... output_dim"]:
        z1 = self.w1(x)
        h1 = swiglu(z1)
        z3 = self.w3(x)
        
        return self.w2(h1 * z3)
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        self.w1.load_state_dict({"weights": state_dict["w1_weight"]})
        self.w2.load_state_dict({"weights": state_dict["w2_weight"]})
        self.w3.load_state_dict({"weights": state_dict["w3_weight"]})