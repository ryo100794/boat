"""Compatibility entry point for the dashboard CLI.

Use ``python -m boatrace_ai.web.dashboard`` for new deployments.
"""

from .web.dashboard import main


if __name__ == "__main__":
    raise SystemExit(main())
