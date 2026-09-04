"""Tests for the Sleeper CLI."""

from __future__ import annotations

from unittest.mock import Mock, patch

import requests
from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def test_rosters_success() -> None:
    """Raw JSON is written to stdout and exit code is 0 on success."""
    fake_response = Mock(spec=requests.Response, ok=True)
    fake_response.status_code = 200
    fake_response.text = '[{"roster_id":1,"owner_id":"123"}]'

    with patch("sleeper.requests.get", return_value=fake_response):
        result = runner.invoke(app, ["rosters", "12345"])

    assert result.exit_code == 0
    assert result.output == '[{"roster_id":1,"owner_id":"123"}]'


def test_rosters_timeout() -> None:
    """Timeout error is written to stderr and exit code is 1."""
    with patch(
        "sleeper.requests.get",
        side_effect=requests.ConnectionError("Connection refused"),
    ):
        result = runner.invoke(app, ["rosters", "12345"])

    assert result.exit_code == 1
    assert "error:" in result.output
    assert "12345" in result.output


def test_rosters_non_200() -> None:
    """Non-200 HTTP response is written to stderr and exit code is 1."""
    fake_response = Mock(spec=requests.Response, ok=False)
    fake_response.status_code = 404

    with patch("sleeper.requests.get", return_value=fake_response):
        result = runner.invoke(app, ["rosters", "12345"])

    assert result.exit_code == 1
    assert "error:" in result.output
    assert "404" in result.output
    assert "12345" in result.output
