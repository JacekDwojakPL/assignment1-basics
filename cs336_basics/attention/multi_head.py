import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange, einsum
from .scaled_attention import scaled_dot_product_attention
from .positional import RotaryPositionEmbeddings
from cs336_basics.nn import Linear

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, device: str = "cpu"):
        super(MultiHeadAttention, self).__init__()
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_q = d_model // num_heads
        self.d_k = self.d_q
        self.d_v = self.d_q
        self.d_o = self.d_q
        self.Q = Linear(self.d_model, self.d_model, device=self.device)
        self.K = Linear(self.d_model, self.d_model, device=self.device)
        self.V = Linear(self.d_model, self.d_model, device=self.device)
        self.O = Linear(self.d_model, self.d_model, device=self.device)
        self.rope = None
        if theta:
            self.rope = RotaryPositionEmbeddings(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=self.device)
   
    def forward(self, in_features: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_model"]:
        seq_length = in_features.shape[1]
        
        q = rearrange(self.Q(in_features), "... seq_len (num_head head_dim) -> ... num_head seq_len head_dim", num_head=self.num_heads)
        k = rearrange(self.K(in_features), "... seq_len (num_head head_dim) -> ... num_head seq_len head_dim", num_head=self.num_heads)
        v = rearrange(self.V(in_features), "... seq_len (num_head head_dim) -> ... num_head seq_len head_dim", num_head=self.num_heads)
        
        if self.rope and token_positions is not None:
            q = self.rope.forward(q, token_positions[0])
            k = self.rope.forward(k, token_positions[0])
        
        
        mask = torch.tril(torch.ones((seq_length, seq_length), device=self.device)) == 1
        
        context = scaled_dot_product_attention(query=q, key=k, value=v, boolean_mask=mask)
        context = rearrange(context, "... num_head seq_len head_dim -> ... seq_len (num_head head_dim)")
        
        return self.O(context)
    
    def load_state_dict(self, state_dict):
        self.Q.load_state_dict({"weights" : state_dict["q_proj_weight"]})
        self.K.load_state_dict({"weights" : state_dict["k_proj_weight"]})
        self.V.load_state_dict({"weights" : state_dict["v_proj_weight"]})
        self.O.load_state_dict({"weights" : state_dict["o_proj_weight"]})