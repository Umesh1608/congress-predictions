"""Tests for signal and alert config model creation."""


class TestSignalModel:
    def test_signal_import(self):
        from src.models.signal import Signal
        assert Signal.__tablename__ == "signal"

    def test_alert_config_import(self):
        from src.models.signal import AlertConfig
        assert AlertConfig.__tablename__ == "alert_config"

    def test_signal_fields(self):
        from src.models.signal import Signal
        # Verify key columns exist
        columns = {c.name for c in Signal.__table__.columns}
        assert "signal_type" in columns
        assert "ticker" in columns
        assert "strength" in columns
        assert "confidence" in columns
        assert "evidence" in columns
        assert "is_active" in columns
        assert "expires_at" in columns
