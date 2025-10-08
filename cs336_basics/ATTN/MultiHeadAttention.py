import torch
from einops import rearrange, einsum
from .ScaledDotProductAttention import scaled_dot_product_attention

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_q = d_model // num_heads
        self.d_k = self.d_q
        self.d_v = self.d_q
        self.d_o = self.d_q
        self.Q = torch.nn.Parameter(torch.randn((self.d_model, self.num_heads, self.d_q)))
        self.K = torch.nn.Parameter(torch.randn((self.d_model, self.num_heads, self.d_k)))
        self.V = torch.nn.Parameter(torch.randn((self.d_model, self.num_heads, self.d_v)))
        self.O = torch.nn.Parameter(torch.randn((self.num_heads, self.d_v, self.d_model)))
   
    def forward(self, in_features):
        _, seq_length, embedding_dim = in_features.shape
        assert self.d_model == embedding_dim, "different shapes"
        assert self.d_q == embedding_dim // self.num_heads
        assert self.d_k == embedding_dim // self.num_heads
        assert self.d_v == embedding_dim // self.num_heads
        q = einsum(in_features, self.Q, "... seq_len embed_dim, embed_dim head head_dim -> ... head seq_len head_dim")
        k = einsum(in_features, self.K, "... seq_len embed_dim, embed_dim head head_dim -> ... head seq_len head_dim")
        v = einsum(in_features, self.V, "... seq_len embed_dim, embed_dim head head_dim -> ... head seq_len head_dim")
        mask = torch.tril(torch.ones((seq_length, seq_length))) == 1
        context = scaled_dot_product_attention(query=q, key=k, value=v, boolean_mask=mask)
        out = einsum(context, self.O, "... head seq_len head_dim, head head_dim out_dim-> ... seq_len out_dim")

        return out
    
    def load_state_dict(self, state_dict):
        self.Q = torch.nn.Parameter(rearrange(state_dict["q_proj_weight"], "embed_dim (heads head_dim) -> embed_dim heads head_dim", heads=self.num_heads))
        self.K = torch.nn.Parameter(rearrange(state_dict["k_proj_weight"], "embed_dim (heads head_dim) -> embed_dim heads head_dim", heads=self.num_heads))
        self.V = torch.nn.Parameter(rearrange(state_dict["v_proj_weight"], "embed_dim (heads head_dim) -> embed_dim heads head_dim", heads=self.num_heads))
        self.O = torch.nn.Parameter(rearrange(state_dict["o_proj_weight"], "embed_dim (heads head_dim) -> heads head_dim embed_dim", heads=self.num_heads))