"""Tests for dataset builder and temporal splitter."""

import pandas as pd

from src.ml.dataset import TemporalSplitter


class TestTemporalSplitter:
    def test_basic_split(self):
        # Create 24 months of data
        dates = pd.date_range("2022-01-01", periods=730, freq="D")
        df = pd.DataFrame({
            "transaction_date": dates,
            "feature": range(len(dates)),
        })
        splitter = TemporalSplitter(n_splits=3, train_months=12, test_months=3)
        splits = splitter.split(df)
        assert len(splits) > 0

    def test_no_future_leakage(self):
        """Ensure test data always comes after train data."""
        dates = pd.date_range("2022-01-01", periods=730, freq="D")
        df = pd.DataFrame({
            "transaction_date": dates,
            "feature": range(len(dates)),
        })
        splitter = TemporalSplitter(n_splits=3, train_months=12, test_months=3)
        splits = splitter.split(df)

        for train_idx, test_idx in splits:
            train_max = df.iloc[train_idx]["transaction_date"].max()
            test_min = df.iloc[test_idx]["transaction_date"].min()
            assert test_min >= train_max

    def test_insufficient_data(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "transaction_date": dates,
            "feature": range(len(dates)),
        })
        splitter = TemporalSplitter(n_splits=3, train_months=12, test_months=3)
        splits = splitter.split(df)
        assert len(splits) == 0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        splitter = TemporalSplitter()
        assert splitter.split(df) == []

    def test_missing_date_column(self):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        splitter = TemporalSplitter()
        assert splitter.split(df) == []
