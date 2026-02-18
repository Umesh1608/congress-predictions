"""Tests for ML feature engineering."""

from datetime import date

from src.ml.features import (
    CHAMBER_MAP,
    FILER_TYPE_MAP,
    PARTY_MAP,
    TRANSACTION_TYPE_MAP,
    _compute_rsi,
    _std,
    member_features,
    network_features_from_dict,
    trade_features,
)


class TestTradeFeatures:
    def test_basic_purchase(self):
        trade = {
            "amount_range_low": 15001,
            "amount_range_high": 50000,
            "transaction_type": "purchase",
            "filer_type": "member",
            "transaction_date": date(2024, 3, 1),
            "disclosure_date": date(2024, 3, 15),
        }
        features = trade_features(trade)
        assert features["amount_midpoint"] == 32500.5
        assert features["is_purchase"] == 1.0
        assert features["tx_direction"] == 1.0
        assert features["filer_type_encoded"] == 0.0
        assert features["disclosure_lag_days"] == 14.0

    def test_sale_by_spouse(self):
        trade = {
            "amount_range_low": 1001,
            "amount_range_high": 15000,
            "transaction_type": "sale",
            "filer_type": "spouse",
        }
        features = trade_features(trade)
        assert features["is_purchase"] == 0.0
        assert features["tx_direction"] == -1.0
        assert features["filer_type_encoded"] == 1.0

    def test_missing_amounts(self):
        trade = {"transaction_type": "exchange"}
        features = trade_features(trade)
        assert features["amount_midpoint"] == 0
        assert features["amount_log"] == 0.0

    def test_missing_dates_default_lag(self):
        trade = {"transaction_type": "purchase"}
        features = trade_features(trade)
        assert features["disclosure_lag_days"] == 30.0


class TestMemberFeatures:
    def test_basic_member(self):
        member = {
            "party": "Democrat",
            "chamber": "house",
            "nominate_dim1": -0.5,
            "nominate_dim2": 0.3,
            "first_elected": 2010,
            "committee_count": 3,
        }
        features = member_features(member)
        assert features["party_encoded"] == 0.0
        assert features["chamber_encoded"] == 0.0
        assert features["nominate_dim1"] == -0.5
        assert features["committee_count"] == 3.0
        assert features["years_in_office"] > 0

    def test_empty_member(self):
        features = member_features({})
        assert features["party_encoded"] == 2.0  # defaults to Independent
        assert features["nominate_dim1"] == 0.0
        assert features["years_in_office"] == 10.0  # default


class TestNetworkFeatures:
    def test_with_connections(self):
        data = {
            "lobbying_connections": 5,
            "has_campaign_donor": True,
            "degree": 12,
            "has_suspicious_triangle": True,
        }
        features = network_features_from_dict(data)
        assert features["lobbying_connection_count"] == 5.0
        assert features["campaign_donor_connection"] == 1.0
        assert features["network_degree"] == 12.0
        assert features["has_lobbying_triangle"] == 1.0

    def test_empty_data(self):
        features = network_features_from_dict({})
        assert features["lobbying_connection_count"] == 0.0
        assert features["campaign_donor_connection"] == 0.0


class TestHelpers:
    def test_std_basic(self):
        assert abs(_std([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.5811) < 0.01

    def test_std_single_value(self):
        assert _std([5.0]) == 0.0

    def test_rsi_neutral(self):
        # Equal ups and downs
        prices = [100, 99, 100, 99, 100, 99, 100, 99, 100, 99, 100, 99, 100, 99, 100]
        rsi = _compute_rsi(prices)
        assert 40 < rsi < 60  # approximately neutral

    def test_rsi_empty(self):
        assert _compute_rsi([]) == 50.0

    def test_rsi_single(self):
        assert _compute_rsi([100]) == 50.0
