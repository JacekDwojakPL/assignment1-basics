import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, device=None, dtype=None):
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        self.mean = 0
        self.std = 2 / (input_dim + output_dim)
        self.w = torch.empty((output_dim, input_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(self.w, self.mean, self.std, -3*self.std, 3*self.std)

    def forward(self, x):
        return x.to(self.device) @ self.w.T
    
    def load_state_dict(self, state_dict):
        self.w = state_dict["weights"]