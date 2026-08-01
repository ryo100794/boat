from pathlib import Path

import pytest

from teleboat_agent.account_balance import AccountBalanceError, parse_available_balance


FIXTURES = Path(__file__).parent / "fixtures" / "teleboat"


def test_20260802_unavailable_balance_response_fails_closed() -> None:
    response = (FIXTURES / "balance-panel-unavailable-20260802.txt").read_text()

    with pytest.raises(AccountBalanceError, match="unavailable"):
        parse_available_balance(response)
