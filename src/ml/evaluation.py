"""Model evaluation metrics for classification and regression."""

from __future__ import annotations

import math

import numpy as np


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute classification metrics.

    Returns dict with accuracy, precision, recall, f1, and optionally auc.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    if n == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    accuracy = float(np.mean(y_true == y_pred))

    # Binary classification metrics
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # AUC-ROC if probabilities provided
    if y_proba is not None:
        metrics["auc"] = _compute_auc(y_true, y_proba)

    return metrics


def evaluate_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics.

    Returns dict with mae, rmse, r2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    if n == 0:
        return {"mae": 0, "rmse": 0, "r2": 0}

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(np.mean(errors**2)))

    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def compute_profit_metrics(
    returns: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute trading-specific metrics.

    Args:
        returns: Actual returns for each trade.
        predictions: Predicted probabilities of profitability.
        threshold: Probability threshold for taking a trade.
    """
    returns = np.asarray(returns, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    # Filter to trades we would have taken
    mask = predictions >= threshold
    if not np.any(mask):
        return {"profit_factor": 0, "win_rate": 0, "avg_return": 0, "trades_taken": 0}

    taken_returns = returns[mask]
    wins = taken_returns[taken_returns > 0]
    losses = taken_returns[taken_returns <= 0]

    total_wins = float(np.sum(wins)) if len(wins) > 0 else 0
    total_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0

    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
    win_rate = len(wins) / len(taken_returns) if len(taken_returns) > 0 else 0
    avg_return = float(np.mean(taken_returns))

    return {
        "profit_factor": profit_factor if profit_factor != float("inf") else 99.99,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "trades_taken": float(len(taken_returns)),
    }


def _compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUC-ROC using the trapezoidal rule."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Sort by descending score
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_idx]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_prev = 0.0
    fpr_prev = 0.0
    auc = 0.0
    tp = 0.0
    fp = 0.0

    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev = tpr
        fpr_prev = fpr

    return float(auc)
