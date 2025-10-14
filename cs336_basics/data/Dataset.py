import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

class Dataset(TorchDataset):
    def __init__(self, data, context_length):
        super(Dataset, self).__init__()
        self.context_length = context_length
        self.data = data
    
    def __getitem__(self, index):
        x = torch.tensor(self.data[index:index+self.context_length], dtype=torch.long)
        y = torch.tensor(self.data[index+1:index+1+self.context_length],  dtype=torch.long)
        return x, y 
    
    def __len__(self):
        return len(self.data) - self.context_length
    

def create_dataloader(data, batch_size, context_length, device="cpu"):
    def collate_fn(batch):
        x_batch = torch.stack([item[0] for item in batch]).to(device)
        y_batch = torch.stack([item[1] for item in batch]).to(device)
        return x_batch, y_batch
    dataset = Dataset(data, context_length)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader