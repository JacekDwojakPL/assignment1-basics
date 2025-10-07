import torch
from jaxtyping import Float, Bool
from einops import einsum
from .Softmax import softmax

def scaled_dot_product_attention(query: Float[torch.Tensor, " ... queries d_k"], 
                                 key: Float[torch.Tensor, " ... keys d_k"], 
                                 value: Float[torch.Tensor, " ... values d_v"], 
                                 boolean_mask: Bool[torch.Tensor, " ... queries keys"] | None = None):
    d_k = torch.tensor(query.shape[-1])
    attention_scores = einsum(query, 
                              key, 
                              "batch_size ... decoder_seq_len d_k, batch_size ... encoder_seq_len d_k -> batch_size ... decoder_seq_len encoder_seq_len")
    attention_scores = attention_scores / torch.sqrt(d_k)
    if boolean_mask is not None:
        attention_scores.masked_fill_(boolean_mask == False, float("-inf"))
    attention_weights = softmax(attention_scores, -1) # shape batch_size ... decoder_seq_len encoder_seq_len
    context = einsum(attention_weights, value, "batch_size ... decoder_seq_len encoder_seq_len, batch_size ... encoder_seq_len d_v -> batch_size ... decoder_seq_len d_v")

    return context