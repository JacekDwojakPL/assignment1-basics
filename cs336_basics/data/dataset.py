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
    

def create_dataloader(data: List[int], batch_size: int, context_length: int, device: str = "cpu") -> DataLoader:
    def collate_fn(batch: List[Tuple[Int[Tensor, "context_length"], Int[Tensor, "context_length"]]]) -> Tuple[Int[Tensor, "batch context_length"], Int[Tensor, "batch context_length"]]:
        x_batch = torch.stack([item[0] for item in batch]).to(device)
        y_batch = torch.stack([item[1] for item in batch]).to(device)
        return x_batch, y_batch
    dataset = Dataset(data, context_length)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader