"""Tests for AutoGluon predictor with synthetic data."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


class TestAutoGluonPredictor:
    def test_predict_without_training_raises(self):
        from src.ml.models.autogluon_predictor import AutoGluonPredictor

        predictor = AutoGluonPredictor()
        try:
            predictor.predict(np.array([[1, 2, 3]]))
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_predict_proba_without_training_raises(self):
        from src.ml.models.autogluon_predictor import AutoGluonPredictor

        predictor = AutoGluonPredictor()
        try:
            predictor.predict_proba(np.array([[1, 2, 3]]))
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_model_name(self):
        from src.ml.models.autogluon_predictor import AutoGluonPredictor

        predictor = AutoGluonPredictor()
        assert predictor.model_name == "autogluon"

    def test_time_limit_default(self):
        from src.ml.models.autogluon_predictor import AutoGluonPredictor

        predictor = AutoGluonPredictor()
        assert predictor.time_limit == 300

    def test_time_limit_custom(self):
        from src.ml.models.autogluon_predictor import AutoGluonPredictor

        predictor = AutoGluonPredictor(time_limit=600)
        assert predictor.time_limit == 600

    def test_dataframe_conversion(self):
        """Verify that numpy arrays are correctly converted to DataFrames."""
        from src.ml.models.autogluon_predictor import AutoGluonPredictor

        predictor = AutoGluonPredictor()
        predictor.feature_columns = ["f1", "f2", "f3"]

        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        col_names = predictor.feature_columns
        df = pd.DataFrame(X, columns=col_names)
        assert list(df.columns) == ["f1", "f2", "f3"]
        assert df.shape == (2, 3)

    @patch("src.ml.models.autogluon_predictor._get_ag")
    def test_train_creates_model(self, mock_get_ag):
        """Test that train() calls TabularPredictor with correct params."""
        from src.ml.models.autogluon_predictor import AutoGluonPredictor

        mock_tp = MagicMock()
        mock_tp.predict.return_value = pd.Series([1, 0, 1])
        mock_tp.predict_proba.return_value = pd.DataFrame({
            0: [0.3, 0.6, 0.2], 1: [0.7, 0.4, 0.8]
        })
        mock_tp.leaderboard.return_value = pd.DataFrame({
            "model": ["LightGBM"], "score_val": [0.8]
        })

        mock_constructor = MagicMock(return_value=mock_tp)
        mock_tp.fit.return_value = mock_tp
        mock_get_ag.return_value = mock_constructor

        predictor = AutoGluonPredictor(time_limit=60)
        predictor.feature_columns = ["f1", "f2"]

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([1, 0, 1])

        metrics = predictor.train(X, y)
        assert isinstance(metrics, dict)
        assert predictor.model is not None

    @patch("src.ml.models.autogluon_predictor._get_ag")
    def test_predict_returns_array(self, mock_get_ag):
        """Test that predict() returns correct numpy array."""
        from src.ml.models.autogluon_predictor import AutoGluonPredictor

        mock_tp = MagicMock()
        mock_tp.predict.return_value = pd.Series([1.0, 0.0])
        mock_get_ag.return_value = MagicMock()

        predictor = AutoGluonPredictor()
        predictor.feature_columns = ["f1", "f2"]
        predictor.model = mock_tp

        result = predictor.predict(np.array([[1, 2], [3, 4]]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    @patch("src.ml.models.autogluon_predictor._get_ag")
    def test_predict_proba_returns_class1(self, mock_get_ag):
        """Test that predict_proba() returns class 1 probabilities."""
        from src.ml.models.autogluon_predictor import AutoGluonPredictor

        mock_tp = MagicMock()
        mock_tp.predict_proba.return_value = pd.DataFrame({
            0: [0.3, 0.7], 1: [0.7, 0.3]
        })
        mock_get_ag.return_value = MagicMock()

        predictor = AutoGluonPredictor()
        predictor.feature_columns = ["f1", "f2"]
        predictor.model = mock_tp

        result = predictor.predict_proba(np.array([[1, 2], [3, 4]]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result, [0.7, 0.3])

    def test_save_and_load_metadata(self):
        """Test save/load creates valid metadata JSON."""
        from src.ml.models.autogluon_predictor import AutoGluonPredictor

        predictor = AutoGluonPredictor(time_limit=120)
        predictor.feature_columns = ["f1", "f2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake AG directory with some content
            ag_dir = str(Path(tmpdir) / "ag_source")
            Path(ag_dir).mkdir()
            (Path(ag_dir) / "dummy.txt").write_text("test")
            predictor._ag_dir = ag_dir

            # Save to a separate output directory
            out_dir = str(Path(tmpdir) / "output")
            Path(out_dir).mkdir()
            path = str(Path(out_dir) / "model.json")

            predictor.save(path)

            # Verify metadata was written
            with open(path) as f:
                metadata = json.load(f)
            assert "ag_dir" in metadata
            assert metadata["feature_columns"] == ["f1", "f2"]
            assert metadata["time_limit"] == 120

    def test_feature_importance_empty(self):
        from src.ml.models.autogluon_predictor import AutoGluonPredictor

        predictor = AutoGluonPredictor()
        assert predictor.get_feature_importance() == {}
