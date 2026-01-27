
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import random
import pickle
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from torch.utils.data import Dataset
class StockDataset(Dataset):
    """Ultra-fast dataset using pre-converted numpy arrays"""

    def __init__(self, ticker_data, samples, window_size=60):
        self.ticker_data = ticker_data
        self.samples = samples
        self.window_size = window_size


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, i = self.samples[idx]
        data = self.ticker_data[ticker]

        # Fast numpy slicing (not pandas!)
        X = data['features'][i - self.window_size:i].copy()

        close_now = data['close'][i - 1]
        close_future = data['close'][i + 29]
        y = float(close_future > close_now)

        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # %% [markdown]
    # ## Load or Create Processed Dataset
    # %%
    CHECKPOINT_FILE = 'dataset_checkpoint.pkl'

    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading preprocessed dataset from {CHECKPOINT_FILE}...")
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)

        ticker_data = checkpoint['ticker_data']
        samples = checkpoint['samples']
        FEATURE_COLS = checkpoint['feature_cols']

        print(f"✓ Loaded {len(samples):,} samples from {len(ticker_data)} tickers")
        print(f"✓ Features: {FEATURE_COLS}")

    else:
        print("No checkpoint found. Processing data from scratch...")

        # Load data
        print("Loading CSV...")
        df = pd.read_csv('data/interim/train_clean_2010.csv')

        # Extract Features
        print("Engineering features...")
        FEATURE_COLS = []
        by_ticker = df.groupby('ticker')

        df['return_1d'] = by_ticker['close'].pct_change()
        FEATURE_COLS.append('return_1d')

        df['log_return'] = np.log(df['close'] / by_ticker['close'].shift(1))
        FEATURE_COLS.append('log_return')

        df['volume_z'] = by_ticker['volume'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )
        FEATURE_COLS.append('volume_z')

        df['hl_range'] = (df['high'] - df['low']) / df['close']
        FEATURE_COLS.append('hl_range')

        df = df.dropna().reset_index(drop=True)
        print(f"Data shape after cleaning: {df.shape}")

        # Convert to numpy arrays for FAST indexing
        print("Converting to numpy arrays...")
        ticker_data = {}
        samples = []
        window_size = 60
        horizon = 29

        for ticker, group in tqdm(df.groupby('ticker'), desc="Processing tickers"):
            group = group.sort_values('date').reset_index(drop=True)
            n = len(group)

            if n < window_size + horizon:
                continue

            # Store as numpy arrays (CRITICAL for speed!)
            ticker_data[ticker] = {
                'features': group[FEATURE_COLS].values.astype(np.float32),
                'close': group['close'].values.astype(np.float32)
            }

            # Generate sample indices
            for i in range(window_size, n - horizon):
                samples.append((ticker, i))

        print(f"✓ Processed {len(samples):,} samples from {len(ticker_data)} tickers")

        # Save checkpoint
        print(f"Saving checkpoint to {CHECKPOINT_FILE}...")
        checkpoint = {
            'ticker_data': ticker_data,
            'samples': samples,
            'feature_cols': FEATURE_COLS,
            'window_size': window_size,
            'horizon': horizon
        }
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint, f)
        print("✓ Checkpoint saved!")


    # %% [markdown]
    # ## Fast Dataset Class (using numpy arrays)
    # %%
    # %%
    print("\nCreating dataset...")
    import time

    start = time.time()
    dataset = StockDataset(ticker_data, samples, window_size=60, horizon=30)
    print(f"Dataset created in {time.time() - start:.2f}s")

    # Test single sample speed
    start = time.time()
    x, y = dataset[0]
    print(f"Single sample access: {time.time() - start:.4f}s (should be <0.001s)")

    # Estimate dataset size
    total_size_gb = (len(dataset) * 60 * len(FEATURE_COLS) * 4) / 1024 ** 3
    print(f"Estimated dataset size: {total_size_gb:.2f} GB")

    # %% [markdown]
    # ## DataLoader Setup
    # %%
    loader = DataLoader(
        dataset,
        batch_size=4096,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3
    )

    print(f"DataLoader ready with {len(loader)} batches")

    # Test first batch
    print("Testing first batch...")
    start = time.time()
    xb, yb = next(iter(loader))
    print(f"First batch loaded in {time.time() - start:.2f}s")
    print(f"Batch shapes: {xb.shape}, {yb.shape}")


    # %% [markdown]
    # ## Model Definition
    # %%
    class GRUClassifier(nn.Module):
        def __init__(self, num_features, hidden_size=256, num_layers=2):
            super().__init__()
            self.gru = nn.GRU(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )
            self.fc1 = nn.Linear(hidden_size, 128)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128, 1)

        def forward(self, x):
            _, h = self.gru(x)
            x = self.fc1(h[-1])
            x = self.relu(x)
            x = self.dropout(x)
            return self.fc2(x).squeeze(1)


    # %% [markdown]
    # ## Setup Training
    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    num_features = len(FEATURE_COLS)
    model = GRUClassifier(num_features, hidden_size=256, num_layers=2).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Mixed precision
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler('cuda')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # %% [markdown]
    # ## Training Loop
    # %%
    EPOCHS = 3

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (xb, yb) in enumerate(progress_bar):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

    # %% [markdown]
    # ## Evaluation
    # %%
    print("\nEvaluating model...")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Evaluating"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                logits = model(xb)

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += yb.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Positive samples: {sum(all_labels) / len(all_labels):.2%}")

    # %% [markdown]
    # ## Save Model
    # %%
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'feature_cols': FEATURE_COLS
    }, 'gru_model.pth')
    print("\nModel saved to 'gru_model.pth'")