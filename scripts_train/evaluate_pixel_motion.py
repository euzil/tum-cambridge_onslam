from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_prediction.pixel_motion_model import SmallPixelMotionUNet


def masked_epe(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    epe = torch.linalg.norm(pred - target, dim=1)
    mask = valid[:, 0] > 0.5
    if mask.any():
        return epe[mask]
    return epe.reshape(-1)[:0]


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned pixel motion against zero-flow baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    data = np.load(args.dataset)
    inputs = torch.from_numpy(np.asarray(data["inputs"], dtype=np.float32))
    flows = torch.from_numpy(np.asarray(data["flows"], dtype=np.float32))
    valids = torch.from_numpy(np.asarray(data["valids"], dtype=np.float32))
    frame_ids = np.asarray(data.get("frame_ids", np.arange(len(inputs))), dtype=np.int32)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SmallPixelMotionUNet(
        in_channels=int(ckpt.get("in_channels", inputs.shape[1])),
        base_channels=int(ckpt.get("base_channels", 32)),
        out_channels=int(ckpt.get("out_channels", flows.shape[1])),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = TensorDataset(inputs, flows, valids, torch.from_numpy(frame_ids))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    learned_all = []
    zero_all = []
    rows = []
    with torch.no_grad():
        for x, y, m, fid in loader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            pred = model(x)
            zero = torch.zeros_like(y)
            learned = masked_epe(pred, y, m).detach().cpu().numpy()
            baseline = masked_epe(zero, y, m).detach().cpu().numpy()
            learned_all.append(learned)
            zero_all.append(baseline)

            pred_epe = torch.linalg.norm(pred - y, dim=1).detach().cpu().numpy()
            zero_epe = torch.linalg.norm(zero - y, dim=1).detach().cpu().numpy()
            valid_np = m[:, 0].detach().cpu().numpy() > 0.5
            for b in range(len(fid)):
                mask = valid_np[b]
                if not np.any(mask):
                    continue
                rows.append(
                    [
                        int(fid[b]),
                        int(mask.sum()),
                        float(pred_epe[b][mask].mean()),
                        float(np.median(pred_epe[b][mask])),
                        float(zero_epe[b][mask].mean()),
                        float(np.median(zero_epe[b][mask])),
                    ]
                )

    learned_arr = np.concatenate(learned_all) if learned_all else np.zeros((0,), dtype=np.float32)
    zero_arr = np.concatenate(zero_all) if zero_all else np.zeros((0,), dtype=np.float32)
    metrics = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "device": device,
        **summarize(learned_arr, "learned_epe_px"),
        **summarize(zero_arr, "zero_epe_px"),
    }
    metrics["learned_vs_zero_gain_percent"] = float(
        (metrics["zero_epe_px_mean"] - metrics["learned_epe_px_mean"])
        / max(metrics["zero_epe_px_mean"], 1e-8)
        * 100.0
    )

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "pixel_motion_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    frame_csv = out_path.with_name(out_path.stem + "_per_frame.csv")
    with frame_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "n_valid",
                "learned_epe_mean_px",
                "learned_epe_median_px",
                "zero_epe_mean_px",
                "zero_epe_median_px",
            ]
        )
        writer.writerows(rows)

    print(f"dataset: {args.dataset}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"learned EPE: {metrics['learned_epe_px_mean']:.6f} px")
    print(f"zero-flow EPE: {metrics['zero_epe_px_mean']:.6f} px")
    print(f"relative gain: {metrics['learned_vs_zero_gain_percent']:+.2f}%")
    print(f"saved metrics: {out_path}")
    print(f"saved per-frame csv: {frame_csv}")


if __name__ == "__main__":
    main()
