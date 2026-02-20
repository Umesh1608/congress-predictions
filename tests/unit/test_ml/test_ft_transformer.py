"""Tests for FT-Transformer predictor with synthetic data."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestFTTransformerPredictor:
    def test_predict_without_training_raises(self):
        from src.ml.models.ft_transformer import FTTransformerPredictor

        predictor = FTTransformerPredictor()
        with pytest.raises(RuntimeError):
            predictor.predict(np.array([[1, 2, 3]]))

    def test_predict_proba_without_training_raises(self):
        from src.ml.models.ft_transformer import FTTransformerPredictor

        predictor = FTTransformerPredictor()
        with pytest.raises(RuntimeError):
            predictor.predict_proba(np.array([[1, 2, 3]]))

    def test_model_name(self):
        from src.ml.models.ft_transformer import FTTransformerPredictor

        assert FTTransformerPredictor.model_name == "ft_transformer"

    def test_default_device_selection(self):
        import torch

        from src.ml.models.ft_transformer import FTTransformerPredictor

        predictor = FTTransformerPredictor()
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert predictor.device == expected

    def test_explicit_cpu_device(self):
        from src.ml.models.ft_transformer import FTTransformerPredictor

        predictor = FTTransformerPredictor(device="cpu")
        assert predictor.device == "cpu"

    def test_normalization(self):
        from src.ml.models.ft_transformer import FTTransformerPredictor

        predictor = FTTransformerPredictor()
        predictor._train_mean = np.array([10.0, 20.0])
        predictor._train_std = np.array([2.0, 5.0])

        X = np.array([[12.0, 25.0], [8.0, 15.0]])
        result = predictor._normalize(X)

        np.testing.assert_array_almost_equal(result[0], [1.0, 1.0], decimal=5)
        np.testing.assert_array_almost_equal(result[1], [-1.0, -1.0], decimal=5)

    def test_normalization_without_stats(self):
        """Without training stats, normalize should return input unchanged."""
        from src.ml.models.ft_transformer import FTTransformerPredictor

        predictor = FTTransformerPredictor()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = predictor._normalize(X)
        np.testing.assert_array_equal(result, X)

    def test_build_model(self):
        """Test that _build_model creates a valid FTTransformer."""
        from src.ml.models.ft_transformer import FTTransformerPredictor

        predictor = FTTransformerPredictor(device="cpu")
        model = predictor._build_model(n_features=10)
        assert model is not None

        # Test forward pass
        import torch
        x = torch.randn(4, 10)
        out = model(x, None)
        assert out.shape == (4, 2)

    def test_train_with_synthetic_data(self):
        """Train on small synthetic data — validates full pipeline."""
        from src.ml.models.ft_transformer import FTTransformerPredictor

        np.random.seed(42)
        n_features = 10
        X_train = np.random.randn(100, n_features)
        y_train = (X_train[:, 0] > 0).astype(float)
        X_val = np.random.randn(30, n_features)
        y_val = (X_val[:, 0] > 0).astype(float)

        predictor = FTTransformerPredictor(
            max_epochs=5, patience=3, batch_size=32, device="cpu"
        )
        predictor.feature_columns = [f"f{i}" for i in range(n_features)]
        metrics = predictor.train(X_train, y_train, X_val, y_val)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert predictor.model is not None

    def test_predict_returns_binary(self):
        """After training, predict should return 0/1 values."""
        from src.ml.models.ft_transformer import FTTransformerPredictor

        np.random.seed(42)
        n = 50
        n_features = 5
        X = np.random.randn(n, n_features)
        y = (X[:, 0] > 0).astype(float)

        predictor = FTTransformerPredictor(
            max_epochs=3, batch_size=32, device="cpu"
        )
        predictor.feature_columns = [f"f{i}" for i in range(n_features)]
        predictor.train(X, y)

        preds = predictor.predict(X)
        assert preds.shape == (n,)
        assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_predict_proba_range(self):
        """predict_proba should return values in [0, 1]."""
        from src.ml.models.ft_transformer import FTTransformerPredictor

        np.random.seed(42)
        n = 50
        n_features = 5
        X = np.random.randn(n, n_features)
        y = (X[:, 0] > 0).astype(float)

        predictor = FTTransformerPredictor(
            max_epochs=3, batch_size=32, device="cpu"
        )
        predictor.feature_columns = [f"f{i}" for i in range(n_features)]
        predictor.train(X, y)

        proba = predictor.predict_proba(X)
        assert proba.shape == (n,)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_save_and_load(self):
        """Test save/load round-trip preserves predictions."""
        from src.ml.models.ft_transformer import FTTransformerPredictor

        np.random.seed(42)
        n_features = 5
        X = np.random.randn(20, n_features)
        y = (X[:, 0] > 0).astype(float)

        predictor = FTTransformerPredictor(
            max_epochs=3, batch_size=32, device="cpu"
        )
        predictor.feature_columns = [f"f{i}" for i in range(n_features)]
        predictor.train(X, y)

        original_proba = predictor.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pkl")
            predictor.save(path)

            loaded = FTTransformerPredictor(device="cpu")
            loaded.load(path)

            loaded_proba = loaded.predict_proba(X)
            np.testing.assert_array_almost_equal(original_proba, loaded_proba, decimal=5)

    def test_sample_weight_support(self):
        """Training with sample weights should not crash."""
        from src.ml.models.ft_transformer import FTTransformerPredictor

        np.random.seed(42)
        n_features = 5
        X = np.random.randn(50, n_features)
        y = (X[:, 0] > 0).astype(float)
        weights = np.random.uniform(0.5, 2.0, size=50)

        predictor = FTTransformerPredictor(
            max_epochs=3, batch_size=32, device="cpu"
        )
        predictor.feature_columns = [f"f{i}" for i in range(n_features)]
        metrics = predictor.train(X, y, sample_weight=weights)
        assert isinstance(metrics, dict)
