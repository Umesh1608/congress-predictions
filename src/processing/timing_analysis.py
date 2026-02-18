"""Timing analysis engine for trade-legislation correlation.

For each trade, this module finds:
1. Committee hearings within ±30 days where the member sits on the committee
2. Bills in the member's committee jurisdiction within ±90 days
3. Sector alignment between traded ticker and committee jurisdiction
4. Disclosure lag analysis (transaction_date vs disclosure_date)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy import and_, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.legislation import Bill, Committee, CommitteeHearing, VoteRecord
from src.models.member import CommitteeAssignment
from src.models.trade import TradeDisclosure

logger = logging.getLogger(__name__)

# Mapping of committee codes to sector keywords for rough sector alignment
# This is intentionally broad — will be refined as the system matures
COMMITTEE_SECTOR_MAP: dict[str, list[str]] = {
    # House committees
    "HSAG": ["agriculture", "farm", "food", "crop"],
    "HSAP": [],  # Appropriations — too broad
    "HSAS": ["defense", "military", "aerospace", "arms"],
    "HSBA": ["bank", "finance", "insurance", "fintech", "credit", "crypto"],
    "HSBU": [],  # Budget — too broad
    "HSED": ["education", "school", "university", "workforce"],
    "HSIF": ["energy", "telecom", "health", "pharma", "biotech", "internet", "media"],
    "HSFA": ["defense", "international", "foreign"],
    "HSHA": ["government", "federal"],
    "HSHM": ["security", "cyber", "border"],
    "HSII": ["native", "water", "land", "mining", "oil", "gas", "energy"],
    "HSJU": ["legal", "tech", "patent", "immigration"],
    "HSPW": ["transport", "infrastructure", "aviation", "rail", "highway", "water"],
    "HSRU": [],  # Rules
    "HSSM": ["small business"],
    "HSSY": ["tech", "science", "space", "nasa", "research", "ai"],
    "HSVR": ["veteran", "health"],
    "HSWM": ["tax", "trade", "social security", "medicare", "health"],
    # Senate committees
    "SSAF": ["agriculture", "farm", "food", "forestry"],
    "SSAP": [],  # Appropriations
    "SSAS": ["defense", "military", "aerospace"],
    "SSBK": ["bank", "finance", "housing", "insurance", "crypto"],
    "SSBU": [],  # Budget
    "SSCM": ["telecom", "tech", "media", "internet", "transport", "science"],
    "SSEG": ["energy", "natural", "oil", "gas", "nuclear", "water"],
    "SSEV": ["environment", "climate", "pollution"],
    "SSFI": ["tax", "trade", "social security", "health", "tariff"],
    "SSFR": ["defense", "foreign", "international"],
    "SSHR": ["health", "pharma", "biotech", "labor", "pension", "education"],
    "SSHI": ["security", "government"],
    "SSGA": ["government", "regulatory"],
    "SSIN": ["intelligence", "cyber", "security"],
    "SSJU": ["legal", "tech", "immigration", "patent"],
    "SSRA": [],  # Rules
    "SSSB": ["small business"],
    "SSVA": ["veteran", "health"],
}


@dataclass
class LegislativeContext:
    """Legislative context surrounding a single trade."""

    trade_id: int
    member_bioguide_id: str | None = None
    transaction_date: date = field(default_factory=date.today)
    disclosure_date: date | None = None
    disclosure_lag_days: int | None = None

    # Nearby hearings the member's committee held
    nearby_hearings: list[dict[str, Any]] = field(default_factory=list)

    # Bills in the member's committee jurisdiction near the trade
    nearby_bills: list[dict[str, Any]] = field(default_factory=list)

    # Member's votes near the trade date
    nearby_votes: list[dict[str, Any]] = field(default_factory=list)

    # Sector alignment flags
    committee_sector_alignment: bool = False
    aligned_committees: list[str] = field(default_factory=list)

    # Timing scores
    min_hearing_distance_days: int | None = None
    min_bill_distance_days: int | None = None
    timing_suspicion_score: float = 0.0


async def analyze_trade_context(
    session: AsyncSession,
    trade_id: int,
    hearing_window_days: int = 30,
    bill_window_days: int = 90,
) -> LegislativeContext | None:
    """Build legislative context for a single trade."""

    # Fetch the trade
    trade = await session.get(TradeDisclosure, trade_id)
    if not trade:
        return None

    tx_date = trade.transaction_date
    member_id = trade.member_bioguide_id

    # Compute disclosure lag
    disclosure_lag = None
    if trade.disclosure_date and trade.transaction_date:
        disclosure_lag = (trade.disclosure_date - trade.transaction_date).days

    ctx = LegislativeContext(
        trade_id=trade_id,
        member_bioguide_id=member_id,
        transaction_date=tx_date,
        disclosure_date=trade.disclosure_date,
        disclosure_lag_days=disclosure_lag,
    )

    if not member_id:
        return ctx

    # Get member's committee assignments
    assignments = await session.execute(
        select(CommitteeAssignment).where(
            CommitteeAssignment.member_bioguide_id == member_id
        )
    )
    member_committees = assignments.scalars().all()
    committee_codes = [a.committee_code for a in member_committees]

    if not committee_codes:
        return ctx

    # 1. Find nearby hearings from the member's committees
    hearing_start = tx_date - timedelta(days=hearing_window_days)
    hearing_end = tx_date + timedelta(days=hearing_window_days)

    hearings_result = await session.execute(
        select(CommitteeHearing).where(
            and_(
                CommitteeHearing.committee_code.in_(committee_codes),
                CommitteeHearing.hearing_date >= hearing_start,
                CommitteeHearing.hearing_date <= hearing_end,
            )
        ).order_by(CommitteeHearing.hearing_date)
    )
    for hearing in hearings_result.scalars().all():
        distance = (hearing.hearing_date - tx_date).days
        ctx.nearby_hearings.append({
            "hearing_id": hearing.id,
            "title": hearing.title,
            "date": hearing.hearing_date.isoformat(),
            "committee_code": hearing.committee_code,
            "distance_days": distance,
            "before_trade": distance < 0,
        })

    if ctx.nearby_hearings:
        ctx.min_hearing_distance_days = min(
            abs(h["distance_days"]) for h in ctx.nearby_hearings
        )

    # 2. Find nearby bills in member's committees
    bill_start = tx_date - timedelta(days=bill_window_days)
    bill_end = tx_date + timedelta(days=bill_window_days)

    # Bills where committees JSONB array contains any of the member's committee codes
    bills_result = await session.execute(
        select(Bill).where(
            and_(
                Bill.introduced_date >= bill_start,
                Bill.introduced_date <= bill_end,
            )
        ).order_by(Bill.introduced_date)
    )
    for bill in bills_result.scalars().all():
        # Check if any of the bill's committees overlap with the member's committees
        bill_committees = bill.committees or []
        bill_committee_codes = {
            c.get("code", "")[:4] for c in bill_committees if isinstance(c, dict)
        }
        member_committee_prefixes = {code[:4] for code in committee_codes}

        if bill_committee_codes & member_committee_prefixes:
            distance = (bill.introduced_date - tx_date).days if bill.introduced_date else None
            ctx.nearby_bills.append({
                "bill_id": bill.bill_id,
                "title": bill.title[:200],
                "introduced_date": bill.introduced_date.isoformat() if bill.introduced_date else None,
                "policy_area": bill.policy_area,
                "distance_days": distance,
                "sponsor_bioguide_id": bill.sponsor_bioguide_id,
                "is_sponsor": bill.sponsor_bioguide_id == member_id,
            })

    if ctx.nearby_bills:
        distances = [abs(b["distance_days"]) for b in ctx.nearby_bills if b["distance_days"] is not None]
        if distances:
            ctx.min_bill_distance_days = min(distances)

    # 3. Find member's votes near the trade
    vote_start = tx_date - timedelta(days=7)
    vote_end = tx_date + timedelta(days=7)

    votes_result = await session.execute(
        select(VoteRecord).where(
            and_(
                VoteRecord.member_bioguide_id == member_id,
                VoteRecord.vote_date >= vote_start,
                VoteRecord.vote_date <= vote_end,
            )
        ).order_by(VoteRecord.vote_date)
    )
    for vote in votes_result.scalars().all():
        ctx.nearby_votes.append({
            "vote_date": vote.vote_date.isoformat(),
            "position": vote.position,
            "question": vote.question,
            "result": vote.result,
        })

    # 4. Check sector alignment between ticker and committee
    ticker = trade.ticker
    if ticker:
        asset_name = (trade.asset_name or "").lower()
        for code in committee_codes:
            prefix = code[:4].upper()
            sector_keywords = COMMITTEE_SECTOR_MAP.get(prefix, [])
            for keyword in sector_keywords:
                if keyword in asset_name:
                    ctx.committee_sector_alignment = True
                    ctx.aligned_committees.append(code)
                    break

    # 5. Compute suspicion score (heuristic, will be replaced by ML in Phase 5)
    ctx.timing_suspicion_score = _compute_suspicion_score(ctx)

    return ctx


def _compute_suspicion_score(ctx: LegislativeContext) -> float:
    """Heuristic suspicion score [0, 1] based on timing patterns.

    This is a rough heuristic that will be replaced by the ML anomaly
    detector in Phase 5. It flags trades that coincide suspiciously
    with legislative events.
    """
    score = 0.0

    # Hearing proximity: closer = more suspicious
    if ctx.min_hearing_distance_days is not None:
        if ctx.min_hearing_distance_days <= 3:
            score += 0.3
        elif ctx.min_hearing_distance_days <= 7:
            score += 0.2
        elif ctx.min_hearing_distance_days <= 14:
            score += 0.1

    # Bill proximity
    if ctx.min_bill_distance_days is not None:
        if ctx.min_bill_distance_days <= 7:
            score += 0.2
        elif ctx.min_bill_distance_days <= 30:
            score += 0.1

    # Sector alignment
    if ctx.committee_sector_alignment:
        score += 0.2

    # Disclosure lag (longer lag = slightly more suspicious)
    if ctx.disclosure_lag_days is not None:
        if ctx.disclosure_lag_days > 40:
            score += 0.15
        elif ctx.disclosure_lag_days > 30:
            score += 0.05

    # Sponsor trading their own bill's sector
    sponsored_bills = [b for b in ctx.nearby_bills if b.get("is_sponsor")]
    if sponsored_bills:
        score += 0.15

    return min(score, 1.0)


async def get_trades_for_committee(
    session: AsyncSession,
    committee_code: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Get all trades by members who sit on a given committee.

    Returns trades annotated with the member's role on the committee.
    """
    result = await session.execute(
        select(TradeDisclosure, CommitteeAssignment.role, CommitteeAssignment.committee_name)
        .join(
            CommitteeAssignment,
            and_(
                TradeDisclosure.member_bioguide_id == CommitteeAssignment.member_bioguide_id,
                CommitteeAssignment.committee_code == committee_code,
            ),
        )
        .order_by(TradeDisclosure.transaction_date.desc())
        .limit(limit)
        .offset(offset)
    )

    trades = []
    for trade, role, committee_name in result.all():
        trades.append({
            "trade_id": trade.id,
            "member_name": trade.member_name,
            "member_bioguide_id": trade.member_bioguide_id,
            "ticker": trade.ticker,
            "asset_name": trade.asset_name,
            "transaction_type": trade.transaction_type,
            "transaction_date": trade.transaction_date.isoformat(),
            "disclosure_date": trade.disclosure_date.isoformat() if trade.disclosure_date else None,
            "amount_range_low": float(trade.amount_range_low) if trade.amount_range_low else None,
            "amount_range_high": float(trade.amount_range_high) if trade.amount_range_high else None,
            "committee_role": role,
            "committee_name": committee_name,
        })

    return trades
