#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import sys

import boatrace_ai.bankroll_optimizer as bankroll


print(
    json.dumps(
        {
            "python": sys.executable,
            "module": bankroll.__file__,
            "signature": str(inspect.signature(bankroll.adaptive_bankroll_streaming)),
            "sys_path": sys.path,
        },
        ensure_ascii=False,
        indent=2,
    )
)
