from __future__ import annotations

import ast

from scripts.update_remote_eval_status import JOBS, build_remote_code


def test_remote_status_generated_code_is_valid_python() -> None:
    source = build_remote_code("root@example.test", "/workspace/boat")

    ast.parse(source)
    assert 'data.get("conditional_order")' in source
    assert 'data.get("bankroll")' in source
    assert 'row["bankroll_gate"]' in source
    assert 'row["bankroll_confidence"]' in source


def test_remote_status_registers_conditional_order_artifact() -> None:
    job = next(row for row in JOBS if row["kind"] == "conditional_order_365d")

    assert job["pid"] == 0
    assert job["output"] == "data/models/conditional_order_365d.json"
    assert job["log"] == "logs/runtime/conditional-order-evaluation.log"
