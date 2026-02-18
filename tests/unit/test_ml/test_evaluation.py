"""Tests for ML evaluation metrics."""

import numpy as np

from src.ml.evaluation import (
    compute_profit_metrics,
    evaluate_classifier,
    evaluate_regressor,
)


class TestClassifierMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        metrics = evaluate_classifier(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        metrics = evaluate_classifier(y_true, y_pred)
        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0

    def test_mixed_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])
        metrics = evaluate_classifier(y_true, y_pred)
        assert 0 < metrics["accuracy"] < 1
        assert metrics["recall"] < 1.0

    def test_auc_with_probabilities(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.2, 0.1])
        metrics = evaluate_classifier(y_true, y_pred, y_proba)
        assert "auc" in metrics
        assert metrics["auc"] > 0.5

    def test_empty_predictions(self):
        metrics = evaluate_classifier(np.array([]), np.array([]))
        assert metrics["accuracy"] == 0


class TestRegressorMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = evaluate_regressor(y_true, y_pred)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0

    def test_imperfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])
        metrics = evaluate_regressor(y_true, y_pred)
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert metrics["r2"] < 1.0

    def test_empty(self):
        metrics = evaluate_regressor(np.array([]), np.array([]))
        assert metrics["mae"] == 0


class TestProfitMetrics:
    def test_profitable_trades(self):
        returns = np.array([0.05, 0.10, -0.02, 0.08])
        predictions = np.array([0.8, 0.9, 0.3, 0.7])
        metrics = compute_profit_metrics(returns, predictions, threshold=0.5)
        assert metrics["win_rate"] > 0
        assert metrics["trades_taken"] == 3  # 3 above 0.5 threshold
        assert metrics["avg_return"] > 0

    def test_no_trades_taken(self):
        returns = np.array([0.05])
        predictions = np.array([0.1])
        metrics = compute_profit_metrics(returns, predictions, threshold=0.5)
        assert metrics["trades_taken"] == 0
        assert metrics["profit_factor"] == 0
