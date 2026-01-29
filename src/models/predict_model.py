import sys

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from src.data.make_torch_datasets import FEATURE_COLS
from src.configs.training_config import TrainingConfig
from src.models.gru_model import GRUModel


def main():
    args = sys.argv
    print("\nLoading test data...")
    test_df = pd.read_csv('data/raw/test.csv')
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    print(f"✓ Test samples: {len(test_df)}")

    checkpoint_path = args[1]
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    cfg_dict = checkpoint["config"]
    cfg = TrainingConfig(**cfg_dict)
    model = None
    model = GRUModel(
            input_size=len(FEATURE_COLS),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    data  = pd.read_csv('data/processed/data.csv')
    ticker_dict = {}
    for ticker, group in tqdm(data.groupby('ticker'), desc="Processing tickers"):
        group = group.sort_values('date').reset_index(drop=True)
        ticker_dict[ticker] = {
            'dates': group['date'].values,
            'features': group[FEATURE_COLS].values.astype(np.float32)
        }
        
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
    
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Pred': predictions
    })

    submission.to_csv('submission.csv', index=False)
    

if __name__ == "__main__":
    main()