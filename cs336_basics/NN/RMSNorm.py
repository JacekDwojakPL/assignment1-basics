import torch
import torch.nn as nn
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weights = torch.empty(d_model)
        nn.init.normal_(self.weights) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_type = x.dtype
        x = x.to(torch.float32)
        mean = self._mean(x)
        result = (x / mean)*self.weights
        
        return result.to(input_type)
        
    def _mean(self, x):
        
        return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
    
    def load_state_dict(self, state_dict):
        self.weights = state_dict["weights"]
