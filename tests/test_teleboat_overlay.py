from teleboat_agent.browser import PlaywrightVoteExecutor


class _Close:
    def __init__(self, visible: bool) -> None:
        self.visible = visible
        self.clicks = []

    def is_visible(self) -> bool:
        return self.visible

    def click(self, *, force: bool) -> None:
        self.clicks.append(force)


class _Closes:
    def __init__(self, closes: list[_Close]) -> None:
        self.closes = closes

    def count(self) -> int:
        return len(self.closes)

    def nth(self, index: int) -> _Close:
        return self.closes[index]


class _Page:
    def __init__(self, closes: list[_Close]) -> None:
        self.closes = _Closes(closes)
        self.waits = []

    def locator(self, selector: str) -> _Closes:
        assert selector == ".modal-close"
        return self.closes

    def wait_for_timeout(self, milliseconds: int) -> None:
        self.waits.append(milliseconds)


def test_dismiss_visible_overlays_closes_only_visible_modals() -> None:
    hidden = _Close(False)
    visible = _Close(True)
    page = _Page([hidden, visible])

    PlaywrightVoteExecutor._dismiss_visible_overlays(page)

    assert hidden.clicks == []
    assert visible.clicks == [True]
    assert page.waits == [500, 200]
