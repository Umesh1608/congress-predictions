"""Tests for prediction service with mocked models."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.ml.predictor import PredictionService


class TestPredictionService:
    def test_initial_state(self):
        service = PredictionService()
        assert service._loaded is False
        assert service._models == {}

    def test_models_dict_empty_by_default(self):
        service = PredictionService()
        assert len(service._models) == 0

    def test_artifact_ids_empty_by_default(self):
        service = PredictionService()
        assert len(service._artifact_ids) == 0

    def test_singleton_pattern(self):
        from src.ml.predictor import get_prediction_service

        # Reset singleton
        import src.ml.predictor
        src.ml.predictor._prediction_service = None

        svc1 = get_prediction_service()
        svc2 = get_prediction_service()
        assert svc1 is svc2

        # Clean up
        src.ml.predictor._prediction_service = None
