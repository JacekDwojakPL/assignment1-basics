import torch
from torch import Tensor
from jaxtyping import Float, Int

def crossentropy(logits: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  
    loss = -logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    return loss.mean()