from torch.utils.data import DataLoader

import torch
from src.data.make_torch_datasets import build_samples, split_samples_time_based
from src.data.stock_dataset import StockDataset
from src.models.gru_model import GRUModel

if __name__ == "__main__":
    samples = build_samples(60)

    train_s, val_s, test_s = split_samples_time_based(samples)

    train_ds = StockDataset(train_s, window_size=60, horizon=30)
    val_ds = StockDataset(val_s, window_size=60, horizon=30)
    test_ds = StockDataset(test_s, window_size=60, horizon=30)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    #X0, y0 = train_ds[0]
    #INPUT_SIZE = X0.shape[1]
    X_batch, y_batch = next(iter(train_loader))

    INPUT_SIZE = X_batch.shape[2]
    SEQ_LEN = X_batch.shape[1]
    BATCH_SIZE = X_batch.shape[0]

    model = GRUModel(
        input_size=INPUT_SIZE,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
    )

    out = model(X_batch)

    print("Input:", X_batch.shape)
    print("Output:", out.shape)