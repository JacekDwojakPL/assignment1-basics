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
        self.Q = torch.nn.Parameter(torch.randn((self.d_model, self.d_model)))
        self.K = torch.nn.Parameter(torch.randn((self.d_model, self.d_model)))
        self.V = torch.nn.Parameter(torch.randn((self.d_model, self.d_model)))
        self.O = torch.nn.Parameter(torch.randn((self.d_model, self.d_model)))
   
    def forward(self, in_features):
        seq_length = in_features.shape[1]
        
        q = einsum(in_features, self.Q, "... embed_dim, out_dim embed_dim -> ... out_dim")
        q = rearrange(q, "... seq_len (num_head head_dim) -> ... num_head seq_len head_dim", num_head=self.num_heads)
        
        k = einsum(in_features, self.K, "... embed_dim, out_dim embed_dim -> ... out_dim")
        k = rearrange(k, "... seq_len (num_head head_dim) -> ... num_head seq_len head_dim", num_head=self.num_heads)
        
        v = einsum(in_features, self.V, "... embed_dim, out_dim embed_dim -> ... out_dim")
        v = rearrange(v, "... seq_len (num_head head_dim) -> ... num_head seq_len head_dim", num_head=self.num_heads)
        
        mask = torch.tril(torch.ones((seq_length, seq_length))) == 1
        
        context = scaled_dot_product_attention(query=q, key=k, value=v, boolean_mask=mask)
        context = rearrange(context, "... num_head seq_len head_dim -> ... seq_len (num_head head_dim)")
        
        out = einsum(context, self.O, "... seq_len embed_dim, out_dim embed_dim-> ... seq_len out_dim")

        return out
    
    def load_state_dict(self, state_dict):
        self.Q.data = state_dict["q_proj_weight"]
        self.K.data = state_dict["k_proj_weight"]
        self.V.data = state_dict["v_proj_weight"]
        self.O.data = state_dict["o_proj_weight"]