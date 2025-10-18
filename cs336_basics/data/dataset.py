import torch
from typing import List, Tuple
from torch import Tensor
from jaxtyping import Int
from torch.utils.data import Dataset as TorchDataset, DataLoader
import numpy as np

class Dataset(TorchDataset):
    def __init__(self, data: List[int], context_length: int):
        super(Dataset, self).__init__()
        self.context_length = context_length
        self.data = data
    
    def __getitem__(self, index: int) -> Tuple[Int[Tensor, "context_length"], Int[Tensor, "context_length"]]:
        x = torch.from_numpy(self.data[index:index+self.context_length].astype(np.int64))
        y = torch.from_numpy(self.data[index+1:index+1+self.context_length].astype(np.int64))
        return x, y

    def __len__(self) -> int:
        return len(self.data) - self.context_length
    

def create_dataloader(data: List[int], context_length: int, device: str = "cpu") -> DataLoader:
    dataset = Dataset(data, context_length)
    
    def get_batch(batch_size):
        indices = torch.randint(0, len(dataset), (batch_size,), device=device)
        x = torch.stack([dataset[i][0].to(device) for i in indices])
        y = torch.stack([dataset[i][1].to(device) for i in indices])
        return (x, y)

    return get_batch