from typing import Dict, List,Optional
from src.data.make_torch_datasets import build_samples, split_samples_time_based
from src.data.stock_dataset import StockDataset
from src.models.gru_model import GRUModel
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    train: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
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

    with context:
        for X, y in loader:
            X = X.to(device)          # (batch, seq_len, input_size)
            y = y.to(device).float()  # (batch,)

            logits = model(X)         # (batch,)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * y.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return {"loss": epoch_loss, "acc": epoch_acc}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float = 1e-3,
):
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            train=True,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            train=False,
            optimizer=None,
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

    return model, history


if __name__ == "__main__":
    torch.manual_seed(42)
    # samples = build_samples(window_size=60)
    # train_s, val_s, test_s = split_samples_time_based(samples)
    #
    # train_ds = StockDataset(train_s, window_size=60, horizon=30)
    # val_ds = StockDataset(val_s, window_size=60, horizon=30)
    # test_ds = StockDataset(test_s, window_size=60, horizon=30)
    #
    # train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    # test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    #
    # #X0, y0 = train_ds[0]
    # #INPUT_SIZE = X0.shape[1]
    # X_batch, y_batch = next(iter(train_loader))
    # input_size = X_batch.shape[2]

    num_train = 512
    num_val = 128
    seq_len = 60
    input_size = 10

    X_train = torch.randn(num_train, seq_len, input_size)
    X_val = torch.randn(num_val, seq_len, input_size)

    y_train = torch.randint(0, 2, (num_train,), dtype=torch.float32)
    y_val = torch.randint(0, 2, (num_val,), dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = GRUModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        bidirectional=False,
    )

    print("Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,
        lr=1e-2,
    )

    print("Training finished.")
