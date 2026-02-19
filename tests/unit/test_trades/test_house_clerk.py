"""Tests for House Clerk PTR scraper collector."""

from datetime import date

from src.ingestion.trades.house_clerk import (
    HouseClerkCollector,
    _parse_ptr_pdf_text,
    _SearchResultParser,
)


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
