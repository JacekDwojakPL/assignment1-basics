import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum, rearrange

class RotaryPositionEmbeddings(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionEmbeddings, self).__init__()
        self.device = device
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.embedding_dim = d_k
        self.k = self.embedding_dim // 2
        self.rotation_matrix  = torch.zeros((self.max_seq_len, self.k, 2, 2), device=self.device)
        for i in range(self.max_seq_len):
            for k in range(self.k):
                phi = torch.tensor(i / self.theta**(2*k/self.embedding_dim), device=self.device)
                s = torch.sin(phi)
                c = torch.cos(phi)
                self.rotation_matrix[i][k] = torch.tensor([[c, s], [-s, c]], device=self.device)
        

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "seq_len"]) -> Float[Tensor, "... seq_len d_k"]:
        angles = self.rotation_matrix[token_positions]
        x_t = rearrange(x, "... (embedding k) -> ... embedding k", k=2)
        rotated = einsum(x_t, angles, "... seq_len embedding k, seq_len embedding k j -> ... seq_len embedding j")
        out = rearrange(rotated, "... embedding k -> ... (embedding k)")

        return out