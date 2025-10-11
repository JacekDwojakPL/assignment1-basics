import torch
from cs336_basics.ATTN import MultiHeadAttention
from cs336_basics.NN import RMSNorm
from cs336_basics.FF import Positionwise

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.mha = MultiHeadAttention(d_model=self.d_model, 
                                      num_heads=self.num_heads,
                                      with_rope=True,
                                      max_seq_len=self.max_seq_len,
                                      theta=self.theta)
        self.ff = Positionwise(input_dim=d_model, output_dim=d_model)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)


    def forward(self, x):
        token_positions = torch.arange(0, x.shape[1]).unsqueeze(0)
        x = self.mha(self.ln1(x), token_positions) + x
        x = self.ff(self.ln2(x)) + x

        return x

    def load_state_dict(self, state_dict):
        self.mha.load_state_dict({"q_proj_weight": state_dict["q_proj_weight"], 
                                  "k_proj_weight": state_dict["k_proj_weight"], 
                                  "v_proj_weight": state_dict["v_proj_weight"], 
                                  "o_proj_weight": state_dict["o_proj_weight"]})
        self.ff.load_state_dict({"w1_weight": state_dict["w1_weight"], 
                                 "w2_weight": state_dict["w2_weight"], 
                                 "w3_weight": state_dict["w3_weight"]})
        self.ln1.load_state_dict({"weights": state_dict["ln1_weights"]})
        self.ln2.load_state_dict({"weights": state_dict["ln2_weights"]})