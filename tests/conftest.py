"""Test configuration and fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_house_trade_raw() -> dict:
    """Sample raw record from House Stock Watcher."""
    return {
        "disclosure_year": 2024,
        "disclosure_date": "01/15/2024",
        "transaction_date": "2024-01-02",
        "owner": "joint",
        "ticker": "NVDA",
        "asset_description": "NVIDIA Corporation",
        "type": "purchase",
        "amount": "$15,001 - $50,000",
        "representative": "Hon. Nancy Pelosi",
        "district": "CA11",
        "ptr_link": "https://example.com/filing.pdf",
        "cap_gains_over_200_usd": "False",
    }


@pytest.fixture
def sample_senate_trade_raw() -> dict:
    """Sample raw record from Senate Stock Watcher."""
    return {
        "transaction_date": "2024-02-10",
        "disclosure_date": "03/01/2024",
        "owner": "Self",
        "ticker": "MSFT",
        "asset_description": "Microsoft Corporation",
        "type": "Purchase",
        "amount": "$1,001 - $15,000",
        "senator": "Tommy Tuberville",
        "ptr_link": "https://example.com/senate_filing.pdf",
    }
