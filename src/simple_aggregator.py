import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from functools import partial
from scipy.special import expit, logsumexp
from sklearn.metrics import roc_auc_score

from utils.datasets import ScanDataset
from utils.CONSTANTS import LABEL_COLS


def aggregate_scan_lse(item: dict, tau: float = 1.0) -> np.ndarray:
    """
    LogSumExp pooling aggregator from patch-level logits
    to scan-level probabilities.

    Args:
        item (dict): One ScanDataset item.
        tau (float): Temperature parameter.
                     Smaller -> closer to max
                     Larger  -> closer to mean

    Returns:
        np.ndarray: Scan-level prediction of shape (14,)
    """
    presence_logits = item["presence_logits"].numpy()  # (N,)
    location_logits = item["location_logits"].numpy()  # (N, 13)

    y_hat = np.zeros(14, dtype=np.float32)

    # ---- locations (0–12) ----
    for l in range(13):
        logits_l = location_logits[:, l]

        # LSE pooling on logits
        pooled_logit = tau * logsumexp(logits_l / tau) - tau * np.log(len(logits_l))
        y_hat[l] = expit(pooled_logit)

    # ---- aneurysm present (13) ----
    pooled_pres = tau * logsumexp(presence_logits / tau) - tau * np.log(
        len(presence_logits)
    )
    y_hat[13] = expit(pooled_pres)

    return y_hat


def aggregate_scan_topk(item: dict, k: int = 3) -> np.ndarray:
    """
    Top-k mean pooling aggregator from patch-level logits
    to scan-level probabilities.

    Args:
        item (dict): One ScanDataset item.
        k (int): Number of top patches to average (k=1 == max pooling).

    Returns:
        np.ndarray: Scan-level prediction of shape (14,)
    """
    presence_logits = item["presence_logits"].numpy()  # (N,)
    location_logits = item["location_logits"].numpy()  # (N, 13)

    y_hat = np.zeros(14, dtype=np.float32)

    # ---- locations (0–12) ----
    for l in range(13):
        logits_l = location_logits[:, l]

        # Guard: fewer than k patches (rare but safe)
        k_eff = min(k, logits_l.shape[0])

        topk_logits = np.partition(logits_l, -k_eff)[-k_eff:]
        pooled_logit = topk_logits.mean()

        y_hat[l] = expit(pooled_logit)

    # ---- aneurysm present (13) ----
    k_eff = min(k, presence_logits.shape[0])
    topk_pres = np.partition(presence_logits, -k_eff)[-k_eff:]
    pooled_pres = topk_pres.mean()

    y_hat[13] = expit(pooled_pres)

    return y_hat


def run_aggregator(
    dataset: ScanDataset, aggregate_scan: callable
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run scan-level aggregation over a dataset and collect predictions / targets.
    """
    y_true = []
    y_pred = []

    for item in tqdm(dataset):
        y_true.append(item["y"].numpy())
        y_pred.append(aggregate_scan(item))

    return np.stack(y_true), np.stack(y_pred)


def mean_weighted_columnwise_auc(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Mean Weighted Columnwise AUROC as defined by the challenge.

    Label 13 (Aneurysm Present) has weight 13, others weight 1.
    """
    aucs = []

    for i in range(y_true.shape[1]):
        # Skip labels with no positive or no negative examples
        if len(np.unique(y_true[:, i])) < 2:
            aucs.append(np.nan)
            continue

        aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))

    aucs = np.array(aucs)

    # weights: 1 for first 13, 13 for last
    weights = np.ones(14)
    weights[13] = 13.0

    weighted_mean_auc = np.nansum(weights * aucs) / np.nansum(weights)

    return weighted_mean_auc, aucs


def list_scan_files(data_paths: list[Path]) -> list[Path]:

    scan_files = []
    for data_path in data_paths:
        scan_files.extend(list(data_path.glob("*.npz")))

    if len(scan_files) == 0:
        raise ValueError(f"No .npz files found in {data_paths}")

    return sorted(scan_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agg_data", type=Path, help="Path to patches folder")

    parser.add_argument(
        "--pool",
        type=str,
        choices=["max", "topk", "lse"],
        default="lse",
        help="Pooling strategy: max, topk, or lse (default: lse)",
    )

    parser.add_argument(
        "--k", type=int, default=3, help="k for top-k pooling (used only if pool=topk)"
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Temperature for LogSumExp pooling (used only if pool=lse)",
    )
    args = parser.parse_args()

    if args.pool == "max":
        aggregate_scan = partial(aggregate_scan_topk, k=1)
        method = "Max Pooling."
    elif args.pool == "topk":
        aggregate_scan = partial(aggregate_scan_topk, k=args.k)
        method = f"Top-{args.k} Pooling."
    elif args.pool == "lse":
        aggregate_scan = partial(aggregate_scan_lse, tau=args.tau)
        method = f"LogSumExp Pooling (tau={args.tau})."
    else:
        raise ValueError(f"Unknown pooling method: {args.pool}")

    data_path = args.agg_data
    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"
    scan_files = list_scan_files([train_path, val_path, test_path])

    scan_dataset = ScanDataset(scan_files)

    print("\nAggregating scan-level predictions & evaluating output...")
    y_true, y_pred = run_aggregator(scan_dataset, aggregate_scan)
    score, aucs = mean_weighted_columnwise_auc(y_true, y_pred)
    print(f"\nMean Weighted AUC for {method}: {score:.4f}")
    print("\nPer-label AUCs:")
    for name, auc in zip(LABEL_COLS, aucs):
        print(f"{name:45s}: {auc:.4f}")
