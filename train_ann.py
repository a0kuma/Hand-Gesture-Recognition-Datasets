import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint_sequential
import wandb


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    if "Gesture" in df.columns:
        label_col = "Gesture"
    else:
        label_col = df.columns[-1]
    labels = df[label_col].astype(str).tolist()
    features = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    unique_labels = sorted(set(labels))
    label_to_idx = {name: i for i, name in enumerate(unique_labels)}
    y = np.array([label_to_idx[name] for name in labels], dtype=np.int64)
    return features, y, unique_labels


def train_val_test_split(
    x: np.ndarray, y: np.ndarray, seed: int, val_frac: float, test_frac: float
) -> SplitData:
    assert 0.0 < val_frac < 1.0
    assert 0.0 < test_frac < 1.0
    assert val_frac + test_frac < 1.0
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]
    n_total = len(x)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val - n_test
    return SplitData(
        x_train=x[:n_train],
        y_train=y[:n_train],
        x_val=x[n_train : n_train + n_val],
        y_val=y[n_train : n_train + n_val],
        x_test=x[n_train + n_val :],
        y_test=y[n_train + n_val :],
    )


def standardize(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        (test_x - mean) / std,
        mean,
        std,
    )


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: list[int],
        dropout: float,
        use_checkpoint: bool,
        checkpoint_segments: int,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.use_checkpoint = use_checkpoint
        self.checkpoint_segments = checkpoint_segments

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint_sequential(self.net, self.checkpoint_segments, x)
        return self.net(x)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ANN on gesture CSV.")
    parser.add_argument("--csv", default="Sequence_Compressed_Dataset.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden", default="512,256,128")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--checkpoint-segments", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y, label_names = load_csv(args.csv)
    split = train_val_test_split(x, y, args.seed, args.val_frac, args.test_frac)
    x_train, x_val, x_test, mean, std = standardize(
        split.x_train, split.x_val, split.x_test
    )

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    model = MLP(
        x_train.shape[1],
        len(label_names),
        hidden,
        args.dropout,
        args.checkpoint,
        args.checkpoint_segments,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(split.y_train)
    )
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(split.y_val))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(split.y_test))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    os.makedirs(args.out_dir, exist_ok=True)
    wandb.init(
        project="hand-gesture-ann",
        config={
            "csv": args.csv,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden": hidden,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "device": str(device),
            "checkpoint": args.checkpoint,
            "checkpoint_segments": args.checkpoint_segments,
        },
    )

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_acc += accuracy(logits, yb).item() * xb.size(0)

        train_loss /= len(train_ds)
        train_acc /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_acc += accuracy(logits, yb).item() * xb.size(0)
        val_loss /= len(val_ds)
        val_acc /= len(val_ds)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_alloc_bytes = torch.cuda.max_memory_allocated()
            peak_res_bytes = torch.cuda.max_memory_reserved()
            peak_alloc_mb = peak_alloc_bytes / (1024 * 1024)
            peak_res_mb = peak_res_bytes / (1024 * 1024)
        else:
            peak_alloc_mb = 0.0
            peak_res_mb = 0.0
        epoch_time = time.perf_counter() - epoch_start
        steps_per_sec = len(train_loader) / max(epoch_time, 1e-8)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "train/epoch_time_sec": epoch_time,
                "train/steps_per_sec": steps_per_sec,
                "train/peak_gpu_mem_alloc_mb": peak_alloc_mb,
                "train/peak_gpu_mem_reserved_mb": peak_res_mb,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            test_loss += loss.item() * xb.size(0)
            test_acc += accuracy(logits, yb).item() * xb.size(0)
    test_loss /= len(test_ds)
    test_acc /= len(test_ds)
    wandb.log({"test/loss": test_loss, "test/acc": test_acc})

    metadata = {
        "label_names": label_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "input_dim": int(x_train.shape[1]),
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Best model saved to: {best_path}")
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
