import torch
from torch import Tensor
from jaxtyping import Float
from typing import Literal
from cs336_basics.attention import MultiHeadAttention
from cs336_basics.nn import RMSNorm
from cs336_basics.feedforward import Positionwise

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 max_seq_len: int, 
                 theta: float | None = None, 
                 device: str = "cpu",
                 normalization: Literal["pre", "post", "off"] = "pre",
                 activation: Literal["silu", "swiglu"] = "swiglu"):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.normalization = normalization
        self.activation = activation
        self.attn = MultiHeadAttention(d_model=self.d_model,
                                       num_heads=self.num_heads,
                                       max_seq_len=self.max_seq_len,
                                       theta=self.theta,
                                       device=self.device)
        self.ffn = Positionwise(input_dim=d_model, output_dim=d_model, device=self.device, activation=self.activation)
        self.ln1 = RMSNorm(d_model=d_model, device=self.device)
        self.ln2 = RMSNorm(d_model=d_model, device=self.device)


    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        token_positions = torch.arange(0, x.shape[1]).unsqueeze(0)
        if self.normalization == 'pre':
            x = self.attn(self.ln1(x), token_positions) + x
            x = self.ffn(self.ln2(x)) + x
        elif self.normalization == 'post':
            x = self.ln1(x + self.attn(x, token_positions))
            x = self.ln2(x + self.ffn(x))
        else:
            x = x + self.attn(x, token_positions)
            x = x + self.ffn(x)
        return x