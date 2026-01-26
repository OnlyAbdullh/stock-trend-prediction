import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):


    def __init__(self, ticker_data, samples, window_size=60, horizon=30):
        self.ticker_data = ticker_data
        self.samples = samples
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, i, y = self.samples[idx]
        data = self.ticker_data[ticker]
        X = data[i - self.window_size +1 :i+1].copy()
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)
