from tqdm import tqdm
from torch.utils.data import Dataset

"""
original_dataset = [
    {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [0, 1]},
    {"input_ids": [3, 4], "attention_mask": [1, 1], "labels": [0, 1]},
    {"input_ids": [5, 6], "attention_mask": [1, 1], "labels": [0, 1]},
    {"input_ids": [7],    "attention_mask": [1],    "labels": [0]},
]
만약 chunk_size가 5 위와 같은 형식의 데이터가 들어온다면

[
    {
        "input_ids": [1, 2, 3, 4, 5],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [0, 1, 0, 1, 0],
    }
]
요렇게 만들어줌
"""

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []

        buffer = {
            "input_ids" : [],
            "attention_mask": [],
            "labels": [],
        }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)