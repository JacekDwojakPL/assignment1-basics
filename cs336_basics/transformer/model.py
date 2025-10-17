import torch
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.nn import Embedding, Linear, RMSNorm
from cs336_basics.transformer import TransformerBlock


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
        self.embedding = Embedding(self.vocab_size, self.d_model, device)
        self.blocks = torch.nn.Sequential(*[TransformerBlock(d_model=self.d_model, 
                                                             num_heads=self.num_heads,
                                                             d_ff=self.d_ff,
                                                             max_seq_len=self.context_length,
                                                             theta=self.theta,
                                                             device=self.device) for _ in range(self.num_layers)])
        self.ln = RMSNorm(d_model=self.d_model, device=self.device)
        self.ff = Linear(self.d_model, self.vocab_size, device=self.device)

    def forward(self, x: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        embeddings = self.embedding(x)
        y = self.blocks(embeddings)
        y = self.ln(y)
        y = self.ff(y)

        return y
    
    def load_state_dict(self, state_dict):
        self.embedding.load_state_dict({"weights": state_dict["token_embeddings.weight"]})
        self.ln.load_state_dict({"weights": state_dict["ln_final.weight"]})
        self.ff.load_state_dict({"weights": state_dict["lm_head.weight"]})
        
        for i in range(self.num_layers):
            state = {"q_proj_weight": state_dict[f"layers.{i}.attn.q_proj.weight"],
                     "k_proj_weight": state_dict[f"layers.{i}.attn.k_proj.weight"],
                     "v_proj_weight": state_dict[f"layers.{i}.attn.v_proj.weight"],
                     "o_proj_weight": state_dict[f"layers.{i}.attn.output_proj.weight"],
                     "w1_weight": state_dict[f"layers.{i}.ffn.w1.weight"],
                     "w2_weight": state_dict[f"layers.{i}.ffn.w2.weight"],
                     "w3_weight": state_dict[f"layers.{i}.ffn.w3.weight"],
                     "ln1_weights": state_dict[f"layers.{i}.ln1.weight"],
                     "ln2_weights": state_dict[f"layers.{i}.ln2.weight"]}
            self.blocks[i].load_state_dict(state)