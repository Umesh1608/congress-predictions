"""Tests for timing analysis suspicion score computation."""

from datetime import date

from src.processing.timing_analysis import LegislativeContext, _compute_suspicion_score


class TestSuspicionScore:
    def test_no_signals_returns_zero(self):
        ctx = LegislativeContext(
            trade_id=1,
            member_bioguide_id="X000001",
            transaction_date=date(2024, 1, 15),
        )
        assert _compute_suspicion_score(ctx) == 0.0

    def test_hearing_within_3_days(self):
        ctx = LegislativeContext(
            trade_id=1,
            member_bioguide_id="X000001",
            transaction_date=date(2024, 1, 15),
            min_hearing_distance_days=2,
        )
        score = _compute_suspicion_score(ctx)
        assert score >= 0.3

    def test_hearing_within_7_days(self):
        ctx = LegislativeContext(
            trade_id=1,
            member_bioguide_id="X000001",
            transaction_date=date(2024, 1, 15),
            min_hearing_distance_days=5,
        )
        score = _compute_suspicion_score(ctx)
        assert 0.2 <= score < 0.3

    def test_sector_alignment_adds_score(self):
        ctx = LegislativeContext(
            trade_id=1,
            member_bioguide_id="X000001",
            transaction_date=date(2024, 1, 15),
            committee_sector_alignment=True,
            aligned_committees=["HSBA00"],
        )
        score = _compute_suspicion_score(ctx)
        assert score >= 0.2

    def test_late_disclosure_adds_score(self):
        ctx = LegislativeContext(
            trade_id=1,
            member_bioguide_id="X000001",
            transaction_date=date(2024, 1, 15),
            disclosure_lag_days=42,
        )
        score = _compute_suspicion_score(ctx)
        assert score >= 0.15

    def test_sponsored_bill_adds_score(self):
        ctx = LegislativeContext(
            trade_id=1,
            member_bioguide_id="X000001",
            transaction_date=date(2024, 1, 15),
            nearby_bills=[{
                "bill_id": "hr1234-118",
                "is_sponsor": True,
                "distance_days": 5,
            }],
            min_bill_distance_days=5,
        )
        score = _compute_suspicion_score(ctx)
        assert score >= 0.35  # bill proximity (0.2) + sponsor (0.15)

    def test_max_score_capped_at_1(self):
        ctx = LegislativeContext(
            trade_id=1,
            member_bioguide_id="X000001",
            transaction_date=date(2024, 1, 15),
            min_hearing_distance_days=1,
            min_bill_distance_days=3,
            committee_sector_alignment=True,
            aligned_committees=["HSBA00"],
            disclosure_lag_days=45,
            nearby_bills=[{"is_sponsor": True, "distance_days": 3}],
        )
        score = _compute_suspicion_score(ctx)
        assert score == 1.0

    def test_bill_proximity_30_days(self):
        ctx = LegislativeContext(
            trade_id=1,
            member_bioguide_id="X000001",
            transaction_date=date(2024, 1, 15),
            min_bill_distance_days=20,
        )
        score = _compute_suspicion_score(ctx)
        assert score >= 0.1
