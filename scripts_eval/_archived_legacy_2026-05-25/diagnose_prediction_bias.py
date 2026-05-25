from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree


def apply_axis_mode(pts: np.ndarray, mode: str) -> np.ndarray:
    out = pts.copy()
    if mode == "none":
        return out
    if mode == "swap_xy":
        return out[:, [1, 0, 2]]
    if mode == "swap_xz":
        return out[:, [2, 1, 0]]
    if mode == "swap_yz":
        return out[:, [0, 2, 1]]
    if mode == "flip_x":
        out[:, 0] *= -1
        return out
    if mode == "flip_y":
        out[:, 1] *= -1
        return out
    if mode == "flip_z":
        out[:, 2] *= -1
        return out
    if mode == "flip_xy":
        out[:, :2] *= -1
        return out
    if mode == "flip_xz":
        out[:, [0, 2]] *= -1
        return out
    if mode == "flip_yz":
        out[:, 1:] *= -1
        return out
    if mode == "flip_xyz":
        out *= -1
        return out
    raise ValueError(f"Unknown mode: {mode}")


def mean_nn(a: np.ndarray, b: np.ndarray, max_samples: int = 8000) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if len(a) > max_samples:
        a = a[np.random.choice(len(a), max_samples, replace=False)]
    if len(b) > max_samples:
        b = b[np.random.choice(len(b), max_samples, replace=False)]
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    return float(np.mean(d))


def evaluate_model(mat_path: Path, axis_mode: str) -> dict[str, float]:
    S = loadmat(mat_path)
    points = np.asarray(S["points"], dtype=np.float64)
    pred = np.asarray(S["predicted_next_points"], dtype=np.float64)
    dyn = np.asarray(S["dynamic"]).reshape(-1).astype(bool)
    fs = np.asarray(S["frame_start_1"]).reshape(-1).astype(int) - 1
    fe = np.asarray(S["frame_end_1"]).reshape(-1).astype(int) - 1

    points = apply_axis_mode(points, axis_mode)
    pred = apply_axis_mode(pred, axis_mode)

    errs_pred_dyn = []
    errs_curr_dyn = []
    motion_norm = []
    n_pairs = 0

    for t in range(len(fs) - 1):
        i0, i1 = fs[t], fe[t]
        j0, j1 = fs[t + 1], fe[t + 1]
        if i1 < i0 or j1 < j0:
            continue

        p_curr = points[i0 : i1 + 1]
        p_pred = pred[i0 : i1 + 1]
        d_curr = dyn[i0 : i1 + 1]
        p_next = points[j0 : j1 + 1]
        d_next = dyn[j0 : j1 + 1]

        curr_dyn = p_curr[d_curr]
        pred_dyn = p_pred[d_curr]
        next_dyn = p_next[d_next]
        if len(curr_dyn) < 20 or len(next_dyn) < 20:
            continue

        e_pred = mean_nn(pred_dyn, next_dyn)
        e_curr = mean_nn(curr_dyn, next_dyn)
        if np.isfinite(e_pred) and np.isfinite(e_curr):
            errs_pred_dyn.append(e_pred)
            errs_curr_dyn.append(e_curr)
            motion_norm.append(float(np.mean(np.linalg.norm(pred_dyn - curr_dyn, axis=1))))
            n_pairs += 1

    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "pred_to_next_dyn_m": float("nan"),
            "curr_to_next_dyn_m": float("nan"),
            "relative_gain_percent": float("nan"),
            "mean_pred_motion_norm_m": float("nan"),
        }

    pred_m = float(np.mean(errs_pred_dyn))
    curr_m = float(np.mean(errs_curr_dyn))
    gain = (curr_m - pred_m) / max(curr_m, 1e-9) * 100.0
    return {
        "n_pairs": n_pairs,
        "pred_to_next_dyn_m": pred_m,
        "curr_to_next_dyn_m": curr_m,
        "relative_gain_percent": float(gain),
        "mean_pred_motion_norm_m": float(np.mean(motion_norm)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose prediction bias against next-frame dynamic observations.")
    ap.add_argument("--model-mat", required=True)
    args = ap.parse_args()

    mat = Path(args.model_mat)
    modes = [
        "none",
        "flip_z",
        "swap_yz",
        "flip_yz",
        "flip_xyz",
    ]
    print(f"model: {mat}")
    print("axis_mode | pairs | pred->next_dyn (m) | curr->next_dyn (m) | gain(%) | pred_motion_norm (m)")
    best = None
    for m in modes:
        s = evaluate_model(mat, m)
        print(
            f"{m:8s} | {int(s['n_pairs']):5d} | {s['pred_to_next_dyn_m']:.4f} | "
            f"{s['curr_to_next_dyn_m']:.4f} | {s['relative_gain_percent']:+7.2f} | "
            f"{s['mean_pred_motion_norm_m']:.4f}"
        )
        if np.isfinite(s["pred_to_next_dyn_m"]):
            if best is None or s["pred_to_next_dyn_m"] < best[1]["pred_to_next_dyn_m"]:
                best = (m, s)

    if best is not None:
        m, s = best
        print("\nBest axis mode by prediction NN error:", m)
        if s["relative_gain_percent"] < 0:
            print("Prediction is worse than using current dynamic points directly (negative gain).")
            print("Likely issue is in prediction model scale/time-step, not only axis transform.")


if __name__ == "__main__":
    main()
