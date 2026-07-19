from boatrace_ai.web_dashboard import HTML


def test_timeline_uses_countdown_only_in_verdict_column() -> None:
    assert "data-deadline-countdown" in HTML
    assert "締切${mm}:${ss}前" in HTML
    assert '${when} <span class="muted">${minLabel(r.minutes_to_deadline)}</span>' not in HTML


def test_countdown_refreshes_each_second() -> None:
    assert "setInterval(updateDeadlineCountdowns,1000)" in HTML
