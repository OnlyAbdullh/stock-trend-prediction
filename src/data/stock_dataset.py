import torch
from torch.utils.data import Dataset
class StockDataset(Dataset):
    ticker_data = {}

    def __init__(self, samples, window_size=60, horizon=30,ticker_data= None):
        self.samples = samples
        self.ticker_data = ticker_data if ticker_data else StockDataset.ticker_data
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, i, y,date = self.samples[idx]
        data = self.ticker_data[ticker]
        X = data[i - self.window_size + 1:i + 1].copy()  # i+1 is excluded
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)
