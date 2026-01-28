from typing import Dict, List, Optional
import os
from datetime import datetime

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.make_torch_datasets import (
    build_samples,
    split_samples_time_based,
    normalize_ticker_data,
)
from src.data.stock_dataset import StockDataset
from src.models.gru_model import GRUModel
from src.configs.training_config import FIRST_CONFIG, SECOND_CONFIG, TrainingConfig
 
CFG = SECOND_CONFIG       
MODE = "resume"            
CHECKPOINT_PATH = r"D:\Stock_trend_project\\models\\gru_unnamed_20260128_115753.pt"     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = torch.cuda.is_available()
if USE_MIXED_PRECISION:
    torch.backends.cudnn.benchmark = True


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    train: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    if train:
        model.train()
        context = torch.enable_grad()
    else:
        model.eval()
        context = torch.no_grad()

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    progress_bar = tqdm(loader, desc="train" if train else "val")

    with context:
        for idx, (X, y) in enumerate(progress_bar):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            if train:
                optimizer.zero_grad()
            if USE_MIXED_PRECISION and train:
                with torch.amp.autocast("cuda"):
                    logits = model(X)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X)
                loss = criterion(logits, y)

                if train:
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * y.size(0)
            if idx % 10 == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return {"loss": epoch_loss, "acc": epoch_acc}


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    cfg: TrainingConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    history: Optional[Dict[str, List[float]]] = None,
    best_val_loss: float = float("inf"),
):
    model.to(device)
    scaler = torch.amp.GradScaler("cuda") if USE_MIXED_PRECISION else None

    criterion = nn.BCEWithLogitsLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    if history is None:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            train=True,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            train=False,
            optimizer=None,
            scaler=None,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state_dict = model.state_dict()

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_acc={train_metrics['acc']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['acc']:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    config_name = getattr(cfg, "name", "unnamed")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gru_{config_name}_{timestamp}.pt"
    save_path = os.path.join(models_dir, filename)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
            "history": history,
            "best_val_loss": best_val_loss,
            "use_mixed_precision": USE_MIXED_PRECISION,
        },
        save_path,
    )
    print(f"\nSaved checkpoint to: {save_path}")

    return model, history


def build_data(cfg: TrainingConfig):
    samples, tickers_data = build_samples(window_size=cfg.window_size, use_cache=False)
    train_s, val_s, test_s = split_samples_time_based(samples)
    tickers_data = normalize_ticker_data(tickers_data, train_s)

    train_ds = StockDataset(
        train_s, window_size=cfg.window_size, horizon=30, ticker_data=tickers_data
    )
    val_ds = StockDataset(
        val_s, window_size=cfg.window_size, horizon=30, ticker_data=tickers_data
    )
    test_ds = StockDataset(
        test_s, window_size=cfg.window_size, horizon=30, ticker_data=tickers_data
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    X_batch, y_batch = next(iter(train_loader))
    input_size = X_batch.shape[2]

    return train_loader, val_loader, test_loader, input_size

if __name__ == "__main__":
    torch.manual_seed(42)

    if MODE == "train":
        cfg = CFG
        train_loader, val_loader, test_loader, input_size = build_data(cfg)

        model = GRUModel(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        )
        print("Starting training from scratch...")
        model, history = train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10,
            cfg=cfg,
        )

    elif MODE == "resume":
        if not CHECKPOINT_PATH:
            raise ValueError("type the CHECKPOINT_PATH")

        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        cfg_dict = ckpt["config"]
        cfg = TrainingConfig(**cfg_dict)

        train_loader, val_loader, test_loader, input_size = build_data(cfg)

        model = GRUModel(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        history_old = ckpt.get("history", None)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

        print(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
        model, history = train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=4,  
            cfg=cfg,
            optimizer=optimizer,
            history=history_old,
            best_val_loss=best_val_loss,
        )

    if torch.cuda.is_available():
        print(
            f"\nPeak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB"
        )
