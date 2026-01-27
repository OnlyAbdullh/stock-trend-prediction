from torch.utils.data import DataLoader

from src.data.make_torch_datasets import build_samples, split_samples_time_based
from src.data.stock_dataset import StockDataset

samples = build_samples(60)

train_s, val_s, test_s = split_samples_time_based(samples)

train_ds = StockDataset(train_s, window_size=60, horizon=30)
val_ds   = StockDataset(val_s,   window_size=60, horizon=30)
test_ds  = StockDataset(test_s,  window_size=60, horizon=30)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
