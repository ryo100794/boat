from types import SimpleNamespace

import pytest

from teleboat_agent.browser import PlaywrightVoteExecutor, VoteExecutionError


class _Node:
    def __init__(self, text: str, visible: bool = True) -> None:
        self.text = text
        self.visible = visible
        self.clicked = False
        self.parent = SimpleNamespace(click=self._click)

    def inner_text(self) -> str:
        return self.text

    def is_visible(self) -> bool:
        return self.visible

    def locator(self, selector: str):
        assert selector == "xpath=parent::*"
        return self.parent

    def _click(self) -> None:
        self.clicked = True


class _Nodes:
    def __init__(self, nodes: list[_Node]) -> None:
        self.nodes = nodes

    def count(self) -> int:
        return len(self.nodes)

    def nth(self, index: int) -> _Node:
        return self.nodes[index]


def test_visible_exact_text_selects_only_unique_current_venue() -> None:
    hidden = _Node("三国", False)
    selected = _Node(" 三国\n", True)
    other = _Node("唐津", True)
