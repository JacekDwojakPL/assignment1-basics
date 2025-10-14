import torch

def save_checkpoint(model, optimizer, iteration, out_filename):
    torch.save({"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), 
                "iteration": iteration}, out_filename)

def load_checkpoint(src_filename, model, optimizer):
    state = torch.load(src_filename)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return state["iteration"]