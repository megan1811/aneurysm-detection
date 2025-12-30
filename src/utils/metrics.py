import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from utils.CONSTANTS import CAT_COLS, LABEL_COLS


def compute_patches_dataset_metrics(df: pd.DataFrame) -> dict:
    """
    Compute and return basic metrics about the patches dataset.

    Args:
        df (pd.DataFrame): DataFrame containing patches metadata.
    Returns:
        dict: Dictionary containing dataset metrics.
    """
    n_total = len(df)
    n_pos = df["label"].sum()
    n_neg = n_total - n_pos

    p_data = (
        n_pos / n_total
    )  # empirical positive patch prevalence (used for sampler weights)
    pos_weight = n_neg / n_pos  # BCEWithLogitsLoss: penalty for false negatives

    df_pos = df[df["label"] == 1]
    loc_labels = df_pos["location"].dropna().values
    loc_labels_int = [LABEL_COLS.index(loc) for loc in loc_labels]

    # Compute sklearn-balanced class weights
    loc_class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CAT_COLS)),
        y=loc_labels_int,
    )
    metrics = {
        "n_total": n_total,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "p_data": p_data,
        "pos_weight": pos_weight,
        "location_class_weights": loc_class_weights.tolist(),
    }

    return metrics
