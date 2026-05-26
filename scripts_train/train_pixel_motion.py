from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_prediction.pixel_motion_model import SmallPixelMotionUNet, masked_smooth_l1


def endpoint_error(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_mask = valid[:, 0] > 0.5
    epe = torch.linalg.norm(pred - target, dim=1)
    if valid_mask.any():
        return epe[valid_mask].mean()
    return epe.mean() * 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small U-Net for dynamic pixel motion prediction.")
    parser.add_argument("--dataset", required=True, help="Path to samples.npz from build_pixel_motion_dataset.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    data = np.load(args.dataset)
    inputs = torch.from_numpy(np.asarray(data["inputs"], dtype=np.float32))
    flows = torch.from_numpy(np.asarray(data["flows"], dtype=np.float32))
    valids = torch.from_numpy(np.asarray(data["valids"], dtype=np.float32))

    n = inputs.shape[0]
    if n < 2:
        raise RuntimeError("Need at least two samples for train/val split.")

    order = rng.permutation(n)
    n_val = max(1, int(round(n * args.val_ratio)))
    n_train = max(1, n - n_val)
    train_idx = torch.as_tensor(order[:n_train], dtype=torch.long)
    val_idx = torch.as_tensor(order[n_train:], dtype=torch.long)
    if len(val_idx) == 0:
        val_idx = train_idx[:1]

    train_ds = TensorDataset(inputs[train_idx], flows[train_idx], valids[train_idx])
    val_ds = TensorDataset(inputs[val_idx], flows[val_idx], valids[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = SmallPixelMotionUNet(
        in_channels=inputs.shape[1],
        base_channels=args.base_channels,
        out_channels=flows.shape[1],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = []
        train_epe = []
        for x, y, m in train_loader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            pred = model(x)
            loss = masked_smooth_l1(pred, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss.append(float(loss.detach().cpu()))
            train_epe.append(float(endpoint_error(pred.detach(), y, m).cpu()))

        model.eval()
        val_loss = []
        val_epe = []
        with torch.no_grad():
            for x, y, m in val_loader:
                x = x.to(device)
                y = y.to(device)
                m = m.to(device)
                pred = model(x)
                val_loss.append(float(masked_smooth_l1(pred, y, m).cpu()))
                val_epe.append(float(endpoint_error(pred, y, m).cpu()))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_loss)),
            "train_epe": float(np.mean(train_epe)),
            "val_loss": float(np.mean(val_loss)),
            "val_epe": float(np.mean(val_epe)),
        }
        history.append(row)

        if row["val_epe"] < best_val:
            best_val = row["val_epe"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_channels": int(inputs.shape[1]),
                    "base_channels": int(args.base_channels),
                    "out_channels": int(flows.shape[1]),
                    "epoch": epoch,
                    "val_epe": best_val,
                    "dataset": str(args.dataset),
                },
                out_dir / "best.pt",
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:04d} "
                f"train_loss={row['train_loss']:.6f} train_epe={row['train_epe']:.4f} "
                f"val_loss={row['val_loss']:.6f} val_epe={row['val_epe']:.4f}"
            )

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"device_used": device, "best_val_epe": best_val}, f, indent=2)

    print(f"saved best checkpoint: {out_dir / 'best.pt'}")
    print(f"best val EPE: {best_val:.6f} px")


if __name__ == "__main__":
    main()
