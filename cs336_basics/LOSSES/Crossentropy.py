import torch

def crossentropy(logits, targets):
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  
    loss = -logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    return loss.mean()