"""Tests for House Clerk PTR scraper collector."""

import asyncio
import time
from datetime import date
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.base import RateLimiter
from src.ingestion.trades.house_clerk import (
    HouseClerkCollector,
    _parse_ptr_pdf_text,
    _SearchResultParser,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _build_fake_search_html(count: int) -> str:
    """Build a fake House clerk search result HTML table."""
    rows = []
    for i in range(count):
        rows.append(
            f'<tr><td><a href="public_disc/ptr-pdfs/2024/{20000000 + i}.pdf">'
            f'Member {i}, Hon.. Test</a></td>'
            f'<td>ST{i:02d}</td><td>2024</td><td>PTR Original</td></tr>'
        )
    return f"<table>{''.join(rows)}</table>"


def _build_minimal_pdf() -> bytes:
    """Build a minimal valid PDF that pypdf can parse."""
    try:
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = BytesIO()
        writer.write(buf)
        return buf.getvalue()
    except ImportError:
        # Fallback: minimal PDF header (won't parse but won't crash tests)
        return b"%PDF-1.4\n"


def _make_mock_response(status_code=200, content=b"", text=""):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code}", request=MagicMock(), response=resp
        )
    return resp


# ---------------------------------------------------------------------------
# Existing tests: HTML parsing, PDF text parsing, transform
# ---------------------------------------------------------------------------

class TestSearchResultParser:
    def test_parse_basic_table(self):
        html = """
        <table>
            <tr>
                <td><a href="public_disc/ptr-pdfs/2025/20032062.pdf">Aderholt, Hon.. Robert B.</a></td>
                <td>AL04</td>
                <td>2025</td>
                <td>PTR Original</td>
            </tr>
            <tr>
                <td><a href="public_disc/financial-pdfs/2025/10071620.pdf">Johnson, Hon.. Mike</a></td>
                <td>LA04</td>
                <td>2025</td>
                <td>FD Original</td>
            </tr>
        </table>
        """
        parser = _SearchResultParser()
        parser.feed(html)
        assert len(parser.rows) == 2
        assert "Aderholt" in parser.rows[0][0]
        assert "ptr-pdfs" in parser.rows[0][0]
        assert parser.rows[0][1] == "AL04"

    def test_empty_table(self):
        html = "<table><thead><tr><th>Name</th></tr></thead></table>"
        parser = _SearchResultParser()
        parser.feed(html)
        assert len(parser.rows) == 0


class TestParsePtrPdfText:
    def test_basic_trade_extraction(self):
        text = """P        T           R
Clerk of the House of Representatives
F     I
Name: Hon. Robert B. Aderholt
Status: Member
State/District:AL04
T
ID Owner Asset Transaction
Type
Date Notification
Date
Amount Cap.
Gains >
$200?
GSK plc American Depositary Shares
(GSK) [ST]
S 07/28/202508/11/2025$1,001 - $15,000
"""
        trades = _parse_ptr_pdf_text(text, "Aderholt, Hon.. Robert B.", "http://example.com/test.pdf")
        assert len(trades) >= 1
        trade = trades[0]
        assert trade["ticker"] == "GSK"
        assert trade["transaction_code"] == "S"
        assert trade["transaction_date_str"] == "07/28/2025"
        assert trade["disclosure_date_str"] == "08/11/2025"
        assert trade["member_name"] == "Aderholt, Hon.. Robert B."

    def test_purchase_trade(self):
        text = """
State/District:CA12
Apple Inc.
(AAPL) [ST]
P 01/15/202502/01/2025$15,001 - $50,000
"""
        trades = _parse_ptr_pdf_text(text, "Pelosi, Nancy", "http://example.com/test.pdf")
        assert len(trades) >= 1
        trade = trades[0]
        assert trade["ticker"] == "AAPL"
        assert trade["transaction_code"] == "P"

    def test_no_trades_in_text(self):
        text = "This is just a cover page with no transaction data."
        trades = _parse_ptr_pdf_text(text, "Test Member", "http://example.com/test.pdf")
        assert len(trades) == 0


class TestHouseClerkTransform:
    def setup_method(self):
        self.collector = HouseClerkCollector(years=[2025])

    def test_purchase_transform(self):
        raw = {
            "member_name": "Hon.. Robert B. Aderholt",
            "ticker": "GSK",
            "asset_name": "GSK plc American Depositary Shares",
            "transaction_code": "P",
            "transaction_date_str": "01/15/2025",
            "disclosure_date_str": "02/01/2025",
            "amount_str": "$1,001 - $15,000",
            "owner": "Self",
            "state_district": "AL04",
            "pdf_url": "http://example.com/test.pdf",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["member_name"] == "Robert B. Aderholt"
        assert result["ticker"] == "GSK"
        assert result["transaction_type"] == "purchase"
        assert result["transaction_date"] == date(2025, 1, 15)
        assert result["chamber"] == "house"
        assert result["source"] == "house_clerk"
        assert result["filer_type"] == "member"

    def test_sale_transform(self):
        raw = {
            "member_name": "Pelosi, Nancy",
            "ticker": "AAPL",
            "asset_name": "Apple Inc.",
            "transaction_code": "S",
            "transaction_date_str": "03/10/2025",
            "disclosure_date_str": "04/01/2025",
            "amount_str": "$50,001 - $100,000",
            "owner": "Spouse",
            "state_district": "CA12",
            "pdf_url": "http://example.com/test.pdf",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["transaction_type"] == "sale"
        assert result["filer_type"] == "spouse"

    def test_no_transaction_date_returns_none(self):
        raw = {
            "member_name": "Test Member",
            "ticker": "AAPL",
            "transaction_code": "P",
            "transaction_date_str": "",
            "disclosure_date_str": "",
            "amount_str": "",
            "owner": "Self",
        }
        result = self.collector.transform(raw)
        assert result is None

    def test_no_member_name_returns_none(self):
        raw = {
            "member_name": "",
            "ticker": "AAPL",
            "transaction_code": "P",
            "transaction_date_str": "01/15/2025",
            "disclosure_date_str": "",
            "amount_str": "",
            "owner": "Self",
        }
        result = self.collector.transform(raw)
        assert result is None

    def test_exchange_type(self):
        raw = {
            "member_name": "Test Member",
            "ticker": "MSFT",
            "asset_name": "Microsoft",
            "transaction_code": "E",
            "transaction_date_str": "01/15/2025",
            "disclosure_date_str": "02/01/2025",
            "amount_str": "$1,001 - $15,000",
            "owner": "Self",
        }
        result = self.collector.transform(raw)
        assert result is not None
        assert result["transaction_type"] == "exchange"

    def test_default_years_is_current(self):
        collector = HouseClerkCollector()
        from datetime import date as d
        assert collector.years == [d.today().year]

    def test_custom_years(self):
        collector = HouseClerkCollector(years=[2020, 2021])
        assert collector.years == [2020, 2021]


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------

class TestConcurrentCollect:
    @pytest.mark.asyncio
    async def test_concurrent_downloads_faster_than_sequential(self):
        """Verify PDFs are downloaded concurrently, not sequentially."""
        fake_html = _build_fake_search_html(count=20)
        fake_pdf = _build_minimal_pdf()

        async def slow_get(url, **kw):
            await asyncio.sleep(0.05)  # 50ms per request
            return _make_mock_response(200, content=fake_pdf)

        collector = HouseClerkCollector(years=[2024])
        collector.rate_limiter = None  # disable rate limiting for this test

        with patch.object(collector.client, "post", new_callable=AsyncMock) as mock_post, \
             patch.object(collector.client, "get", side_effect=slow_get):
            mock_post.return_value = _make_mock_response(200, text=fake_html)

            start = time.monotonic()
            await collector.collect()
            elapsed = time.monotonic() - start

        # 20 requests at 50ms each:
        # Sequential = 1.0s minimum
        # Concurrent (10 slots) = ~0.1s (2 batches of 10)
        assert elapsed < 0.5, f"Expected concurrent execution, took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_malformed_pdf_does_not_crash_batch(self):
        """A 404 or corrupt PDF should not abort the entire batch."""
        fake_html = _build_fake_search_html(count=5)
        fake_pdf = _build_minimal_pdf()
        call_count = 0

        async def mixed_get(url, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # 404 response
                return _make_mock_response(404)
            elif call_count == 3:
                # Not a PDF (HTML error page)
                return _make_mock_response(200, content=b"<!DOCTYPE html>")
            else:
                return _make_mock_response(200, content=fake_pdf)

        collector = HouseClerkCollector(years=[2024])
        collector.rate_limiter = None

        with patch.object(collector.client, "post", new_callable=AsyncMock) as mock_post, \
             patch.object(collector.client, "get", side_effect=mixed_get):
            mock_post.return_value = _make_mock_response(200, text=fake_html)

            # Should not raise — errors are handled per-filing
            trades = await collector.collect()

        assert isinstance(trades, list)
        assert call_count == 5  # all 5 PDFs were attempted

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Verify semaphore limits max concurrent downloads."""
        fake_html = _build_fake_search_html(count=15)
        fake_pdf = _build_minimal_pdf()
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_get(url, **kw):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.02)
            async with lock:
                current_concurrent -= 1
            return _make_mock_response(200, content=fake_pdf)

        collector = HouseClerkCollector(years=[2024])
        collector.rate_limiter = None

        with patch.object(collector.client, "post", new_callable=AsyncMock) as mock_post, \
             patch.object(collector.client, "get", side_effect=tracking_get):
            mock_post.return_value = _make_mock_response(200, text=fake_html)
            await collector.collect()

        # _MAX_CONCURRENT_DOWNLOADS = 10, so max_concurrent should be <= 10
        assert max_concurrent <= 10


class TestRateLimiterConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_acquires_respect_limit(self):
        """Multiple concurrent acquires should not exceed max_calls per period."""
        limiter = RateLimiter(max_calls=3, period_seconds=1.0)
        timestamps: list[float] = []

        async def acquire_and_record():
            await limiter.acquire()
            timestamps.append(time.monotonic())

        # Fire 6 acquires concurrently
        await asyncio.gather(*[acquire_and_record() for _ in range(6)])

        assert len(timestamps) == 6
        # First 3 should be near-instant, rest should be throttled
        # Check no more than 3+1 (tolerance) in any 1-second window
        for t in timestamps:
            window_count = sum(1 for t2 in timestamps if t <= t2 < t + 1.0)
            assert window_count <= 4, f"Too many calls in 1s window: {window_count}"
