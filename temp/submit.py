import pandas as pd
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


# ============================================================
# MODEL DEFINITION (must match training!)
# ============================================================
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


# ============================================================
# LOAD MODEL
# ============================================================
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('gru_model.pth', map_location=device)

FEATURE_COLS = checkpoint['feature_cols']
num_features = len(FEATURE_COLS)

model = GRUClassifier(num_features, hidden_size=256, num_layers=2).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded on {device}")
print(f"✓ Features: {FEATURE_COLS}")

# ============================================================
# LOAD AND PREPROCESS ALL DATA ONCE
# ============================================================
print("\nLoading historical data...")
historical_df = pd.read_csv('../data/interim/train_clean_2010.csv')
historical_df['date'] = pd.to_datetime(historical_df['date'])
print(f"✓ Historical data: {historical_df.shape}")

print("Engineering features...")
by_ticker = historical_df.groupby('ticker')

historical_df['return_1d'] = by_ticker['close'].pct_change()
historical_df['log_return'] = np.log(historical_df['close'] / by_ticker['close'].shift(1))
historical_df['volume_z'] = by_ticker['volume'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-6)
)
historical_df['hl_range'] = (historical_df['high'] - historical_df['low']) / historical_df['close']
historical_df = historical_df.dropna().reset_index(drop=True)

# Convert to numpy arrays grouped by ticker (FAST!)
print("Converting to numpy arrays by ticker...")
ticker_dict = {}
for ticker, group in tqdm(historical_df.groupby('ticker'), desc="Processing tickers"):
    group = group.sort_values('date').reset_index(drop=True)
    ticker_dict[ticker] = {
        'dates': group['date'].values,
        'features': group[FEATURE_COLS].values.astype(np.float32)
    }

print(f"✓ Preprocessed {len(ticker_dict)} tickers")

# ============================================================
# LOAD TEST DATA
# ============================================================
print("\nLoading test data...")
test_df = pd.read_csv('../data/raw/test.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'])
print(f"✓ Test samples: {len(test_df)}")

# ============================================================
# GENERATE PREDICTIONS (FAST!)
# ============================================================
print("\nGenerating predictions...")

window_size = 60
predictions = []
failed_tickers = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
    ticker_id = row['ID']
    target_date = row['Date']

    # Check if ticker exists in our data
    if ticker_id not in ticker_dict:
        failed_tickers.append((ticker_id, "not_found"))
        predictions.append(1)
        continue

    ticker_data = ticker_dict[ticker_id]
    dates = ticker_data['dates']
    features = ticker_data['features']

    # Find the last index before target_date
    mask = dates < target_date
    if mask.sum() < window_size:
        failed_tickers.append((ticker_id, "insufficient_data"))
        predictions.append(1)
        continue

    # Get indices of data before target date
    valid_indices = np.where(mask)[0]

    # Take last 60 indices
    last_60_indices = valid_indices[-window_size:]

    # Extract features (already numpy array!)
    X = features[last_60_indices].copy()

    # Convert to tensor and add batch dimension
    X_tensor = torch.from_numpy(X).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        with torch.amp.autocast('cuda'):  # Using your working syntax
            logits = model(X_tensor)

        prob = torch.sigmoid(logits).item()
        pred = 1 if prob > 0.5 else 0

    predictions.append(pred)

print(f"\n✓ Predictions complete!")
print(f"✓ Failed predictions: {len(failed_tickers)}")
if failed_tickers:
    print(f"  Not found: {sum(1 for _, reason in failed_tickers if reason == 'not_found')}")
    print(f"  Insufficient data: {sum(1 for _, reason in failed_tickers if reason == 'insufficient_data')}")

# ============================================================
# CREATE SUBMISSION FILE
# ============================================================
print("\nCreating submission file...")

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Pred': predictions
})

submission.to_csv('submission.csv', index=False)

print(f"✓ Submission saved to 'submission.csv'")
print(f"✓ Total predictions: {len(predictions)}")
print(f"✓ Positive predictions: {sum(predictions)} ({sum(predictions) / len(predictions) * 100:.1f}%)")

# Show sample
print("\nSample predictions:")
print(submission.head(10))