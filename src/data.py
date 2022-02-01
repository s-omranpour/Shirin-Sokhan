import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split

def count_tokens(x):
    return len(x.replace('<sep>', '0').replace('<|startoftext|>', '1'))


class PoemDataset(Dataset):
    def __init__(self, tokenizer, json_path, window=256):
        self.tokenizer = tokenizer
        self.window = window - 1
        self.samples = json.load(open(json_path))
        self.lens = [count_tokens(samp) - self.window for samp in self.samples]
        self.cum_lens = [0] + [sum(self.lens[:i+1]) for i in range(len(self.samples))]
        
    def __len__(self):
        return self.cum_lens[-1]
    
    def parse_idx(self, idx):
        for i, cl in enumerate(self.cum_lens):
            if idx < cl:
                return i-1, idx - self.cum_lens[i-1]
        return -1, -1

    def __getitem__(self, idx):
        ind, offset = self.parse_idx(idx)
        sample = self.samples[ind].replace('<sep>', '0').replace('<|startoftext|>', '1')
        text = '<s>' + sample[offset:offset+self.window] + '</s>'
        return text.replace('0', '<sep>').replace('1', '<|startoftext|>')
    
    def collate(self, batch):
        return self.tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True)

    
def get_dataloaders(dataset, max_len=256, batch_size=32, val_frac=0.1):
    n = len(dataset)
    v = int(n*val_frac)
    train_dataset, val_dataset = random_split(dataset, [n - v, v])
    print('train dataset has {} samples and val dataset has {} samples'.format(n-v, v))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=dataset.collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate)
    return train_loader, val_loader