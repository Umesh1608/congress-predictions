"""Tests for the trade screener — composite scoring, allocation, and freshness."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.ml.screener import (
    compute_allocations,
    compute_composite_score,
    compute_count_bonus,
    compute_freshness,
)
from src.schemas.recommendations import (
    RecommendationEvidence,
    RiskContext,
    ScreenerResponse,
    TickerRecommendation,
)


# ---------------------------------------------------------------------------
# TestCompositeScore
# ---------------------------------------------------------------------------


class TestCompositeScore:
    """Tests for compute_composite_score pure function."""

    def test_equal_half_inputs(self):
        """All 0.5 inputs → weighted average = 0.5."""
        score = compute_composite_score(0.5, 0.5, 0.5, 0.5, 0.5)
        assert score == pytest.approx(0.5, abs=0.001)

    def test_all_zeros(self):
        score = compute_composite_score(0.0, 0.0, 0.0, 0.0, 0.0)
        assert score == 0.0

    def test_all_ones(self):
        score = compute_composite_score(1.0, 1.0, 1.0, 1.0, 1.0)
        assert score == 1.0

    def test_weights_sum_to_one(self):
        """Weights: 0.30 + 0.25 + 0.20 + 0.15 + 0.10 = 1.0."""
        # If all inputs are 1.0, result should be exactly 1.0
        score = compute_composite_score(1.0, 1.0, 1.0, 1.0, 1.0)
        assert score == pytest.approx(1.0, abs=0.001)

    def test_strength_dominant(self):
        """Strength has highest weight (0.30)."""
        high_strength = compute_composite_score(1.0, 0.0, 0.0, 0.0, 0.0)
        high_ml = compute_composite_score(0.0, 1.0, 0.0, 0.0, 0.0)
        assert high_strength == pytest.approx(0.30, abs=0.001)
        assert high_ml == pytest.approx(0.25, abs=0.001)
        assert high_strength > high_ml


# ---------------------------------------------------------------------------
# TestAllocationComputation
# ---------------------------------------------------------------------------


class TestAllocationComputation:
    """Tests for compute_allocations."""

    def test_equal_scores_equal_allocation(self):
        recs = [
            {"composite_score": 0.5, "current_price": 100.0},
            {"composite_score": 0.5, "current_price": 100.0},
        ]
        result = compute_allocations(recs, 2000.0)
        assert len(result) == 2
        assert result[0]["allocation_pct"] == pytest.approx(0.5, abs=0.01)
        assert result[1]["allocation_pct"] == pytest.approx(0.5, abs=0.01)

    def test_suggested_amount_matches(self):
        recs = [{"composite_score": 1.0, "current_price": 50.0}]
        result = compute_allocations(recs, 1000.0)
        assert result[0]["suggested_amount"] == pytest.approx(1000.0, abs=0.01)

    def test_shares_floor(self):
        recs = [{"composite_score": 1.0, "current_price": 33.0}]
        result = compute_allocations(recs, 100.0)
        # 100 / 33 = 3.03 → floor = 3
        assert result[0]["suggested_shares"] == 3

    def test_zero_price_zero_shares(self):
        recs = [{"composite_score": 1.0, "current_price": 0}]
        result = compute_allocations(recs, 1000.0)
        assert result[0]["suggested_shares"] == 0

    def test_none_price_zero_shares(self):
        recs = [{"composite_score": 1.0, "current_price": None}]
        result = compute_allocations(recs, 1000.0)
        assert result[0]["suggested_shares"] == 0

    def test_min_clamp_10_pct(self):
        """Tiny score should still get at least 10% after clamp."""
        recs = [
            {"composite_score": 0.01, "current_price": 100.0},
            {"composite_score": 0.99, "current_price": 100.0},
        ]
        result = compute_allocations(recs, 1000.0)
        # After clamp: 0.01 gets clamped up to 10%, 0.99 gets clamped to 40%
        # Then renormalized: 10/(10+40) = 0.2, 40/(10+40) = 0.8
        assert result[0]["allocation_pct"] >= 0.10
        assert result[1]["allocation_pct"] <= 0.85

    def test_max_clamp_40_pct(self):
        """Very dominant score gets clamped to 40% before renorm."""
        recs = [
            {"composite_score": 0.95, "current_price": 100.0},
            {"composite_score": 0.05, "current_price": 100.0},
        ]
        result = compute_allocations(recs, 1000.0)
        # Raw: 95/100=0.95 clamped to 0.40, 5/100=0.05 clamped to 0.10
        # Renorm: 0.40/0.50=0.80, 0.10/0.50=0.20
        assert result[0]["allocation_pct"] == pytest.approx(0.80, abs=0.01)
        assert result[1]["allocation_pct"] == pytest.approx(0.20, abs=0.01)

    def test_empty_list(self):
        result = compute_allocations([], 1000.0)
        assert result == []

    def test_allocations_sum_to_one(self):
        recs = [
            {"composite_score": 0.8, "current_price": 50.0},
            {"composite_score": 0.6, "current_price": 100.0},
            {"composite_score": 0.4, "current_price": 200.0},
        ]
        result = compute_allocations(recs, 5000.0)
        total = sum(r["allocation_pct"] for r in result)
        assert total == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# TestFreshnessScore
# ---------------------------------------------------------------------------


class TestFreshnessScore:
    """Tests for compute_freshness."""

    def test_brand_new(self):
        now = datetime.now(timezone.utc)
        score = compute_freshness(now)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_week_old(self):
        week_ago = datetime.now(timezone.utc) - timedelta(hours=168)
        score = compute_freshness(week_ago)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_half_week(self):
        half_week = datetime.now(timezone.utc) - timedelta(hours=84)
        score = compute_freshness(half_week)
        assert score == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# TestCountBonus
# ---------------------------------------------------------------------------


class TestCountBonus:
    """Tests for compute_count_bonus."""

    def test_one_signal(self):
        assert compute_count_bonus(1) == pytest.approx(0.2, abs=0.001)

    def test_five_signals(self):
        assert compute_count_bonus(5) == pytest.approx(1.0, abs=0.001)

    def test_ten_signals_capped(self):
        assert compute_count_bonus(10) == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# TestScreenerResponse
# ---------------------------------------------------------------------------


class TestScreenerResponse:
    """Tests for the Pydantic response schema."""

    def test_disclaimer_present(self):
        resp = ScreenerResponse(
            portfolio_size=3000,
            num_recommendations=0,
            total_investable=0,
            cash_remainder=3000,
            recommendations=[],
            generated_at="2026-02-19T00:00:00Z",
        )
        assert "NOT financial advice" in resp.disclaimer

    def test_recommendation_action_buy(self):
        rec = TickerRecommendation(
            ticker="NVDA",
            composite_score=0.8,
            allocation_pct=0.5,
            suggested_amount=1500,
            suggested_shares=10,
        )
        assert rec.action == "BUY"
        assert rec.evidence.signal_count == 0
        assert rec.risk.volatility_21d is None
