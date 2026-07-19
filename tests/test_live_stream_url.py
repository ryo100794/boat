from __future__ import annotations

from boatrace_ai.web_dashboard import boatcast_live_player_url


def test_live_stream_uses_direct_official_player_url() -> None:
    url = boatcast_live_player_url("01")
    assert url is not None
    assert url.startswith("https://front.player.boatrace-cdn.jp/player/live?")
    assert "stadium=01kiryu" in url
    assert "autoplay=1" in url


def test_unknown_venue_has_no_stream_url() -> None:
    assert boatcast_live_player_url("99") is None
