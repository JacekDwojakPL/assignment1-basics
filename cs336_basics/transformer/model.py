import torch
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.nn import Embedding, Linear, RMSNorm
from cs336_basics.transformer import TransformerBlock
from cs336_basics.attention import softmax

class TransformerModel(torch.nn.Module):
    def __init__(self, vocab_size: int,
                       context_length: int,
                       d_model: int,       
                       num_layers: int,
                       num_heads: int, 
                       d_ff: int,  
                       rope_theta: float | None = None,
                       device: str = "cpu"):
        super(TransformerModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.theta = rope_theta
        self.device = device
        self.token_embeddings = Embedding(self.vocab_size, self.d_model, device)
        self.layers = torch.nn.Sequential(*[TransformerBlock(d_model=self.d_model,
                                                              num_heads=self.num_heads,
                                                              d_ff=self.d_ff,
                                                              max_seq_len=self.context_length,
                                                              theta=self.theta,
                                                              device=self.device) for _ in range(self.num_layers)])
        self.ln_final = RMSNorm(d_model=self.d_model, device=self.device)
        self.lm_head = Linear(self.d_model, self.vocab_size, device=self.device)

    def forward(self, x: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        embeddings = self.token_embeddings(x)
        y = self.layers(embeddings)
        y = self.ln_final(y)
        y = self.lm_head(y)

        return y
    
    def generate(self, start_seq: Int[Tensor, "... seq_len"], max_seq_length: Int = 10, temperature: Float = 0.0):
        temp = torch.tensor(max(temperature, 1e-7), device=self.device, dtype=torch.float64)

        for _ in range(max_seq_length - len(start_seq)):
            z = self(start_seq)
            probs = softmax(z[:, -1, :] / temp, dim=-1)
            idx = torch.multinomial(probs, 1)
            start_seq = torch.cat((start_seq, idx), -1)

        return start_seq.squeeze().numpy()