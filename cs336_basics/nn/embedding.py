import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

class Embedding(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, device=None, dtype=None):
        super(Embedding, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype=dtype
        self.mean = 0
        self.std = 2 / (input_dim + embedding_dim)
        self.weight = nn.Parameter(torch.zeros((input_dim, embedding_dim), device=self.device, dtype=self.dtype))
        nn.init.trunc_normal_(self.weight, self.mean, self.std, -3*self.std, 3*self.std)

    def forward(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.weight[x].to(self.dtype)