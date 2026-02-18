"""Tests for ML models with synthetic data (no real model loading)."""

from unittest.mock import MagicMock, patch

import numpy as np


class TestTradePredictor:
    @patch("src.ml.models.trade_predictor._get_lgbm")
    def test_train_and_predict(self, mock_lgbm):
        from src.ml.models.trade_predictor import TradePredictor

        # Create mock LightGBM classifier
        mock_clf = MagicMock()
        mock_clf.predict.return_value = np.array([1, 0, 1])
        mock_clf.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
        mock_clf.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_lgbm.return_value.LGBMClassifier.return_value = mock_clf

        predictor = TradePredictor()
        predictor.model = mock_clf
        predictor.feature_columns = ["f1", "f2", "f3"]

        preds = predictor.predict(np.array([[1, 2, 3]]))
        assert preds is not None

        proba = predictor.predict_proba(np.array([[1, 2, 3]]))
        assert proba is not None

    def test_predict_without_training_raises(self):
        from src.ml.models.trade_predictor import TradePredictor

        predictor = TradePredictor()
        try:
            predictor.predict(np.array([[1, 2, 3]]))
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_feature_importance_empty(self):
        from src.ml.models.trade_predictor import TradePredictor

        predictor = TradePredictor()
        assert predictor.get_feature_importance() == {}


class TestReturnPredictor:
    @patch("src.ml.models.return_predictor._get_xgb")
    def test_train_and_predict(self, mock_xgb):
        from src.ml.models.return_predictor import ReturnPredictor

        mock_reg = MagicMock()
        mock_reg.predict.return_value = np.array([0.05, -0.02, 0.10])
        mock_reg.feature_importances_ = np.array([0.4, 0.35, 0.25])
        mock_xgb.return_value.XGBRegressor.return_value = mock_reg

        predictor = ReturnPredictor()
        predictor.model = mock_reg

        preds = predictor.predict(np.array([[1, 2, 3]]))
        assert len(preds) == 3

        # predict_proba for regressor returns same as predict
        proba = predictor.predict_proba(np.array([[1, 2, 3]]))
        assert len(proba) == 3


class TestAnomalyDetector:
    def test_filter_features(self):
        from src.ml.models.anomaly_model import AnomalyDetector

        features = {
            "amount_midpoint": 10000,
            "price_change_5d": 0.05,
            "timing_suspicion_score": 0.8,
            "volatility_21d": 0.02,
        }
        filtered = AnomalyDetector.filter_features(features)
        assert "price_change_5d" not in filtered
        assert "volatility_21d" not in filtered
        assert "amount_midpoint" in filtered
        assert "timing_suspicion_score" in filtered

    @patch("src.ml.models.anomaly_model.AnomalyDetector.train")
    def test_predict_proba_rescaling(self, mock_train):
        from src.ml.models.anomaly_model import AnomalyDetector

        detector = AnomalyDetector()
        mock_model = MagicMock()
        mock_model.decision_function.return_value = np.array([-0.5, 0.0, 0.5])
        mock_model.predict.return_value = np.array([-1, 1, 1])
        detector.model = mock_model

        scores = detector.predict_proba(np.array([[1], [2], [3]]))
        # Most negative decision function → highest anomaly score
        assert scores[0] > scores[2]


class TestEnsembleModel:
    def test_build_meta_features(self):
        from src.ml.models.ensemble import EnsembleModel

        meta = EnsembleModel.build_meta_features(
            trade_proba=0.8,
            return_score=0.05,
            anomaly_score=0.3,
            timing_suspicion=0.6,
            avg_sentiment=0.1,
        )
        assert meta.shape == (1, 5)
        assert meta[0, 0] == 0.8
        assert meta[0, 4] == 0.1

    def test_meta_features_names(self):
        from src.ml.models.ensemble import EnsembleModel

        assert len(EnsembleModel.META_FEATURES) == 5
        assert "trade_predictor_proba" in EnsembleModel.META_FEATURES
