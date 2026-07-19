"""Compatibility entry point for the realtime collector."""

from .runtime.collector import main


if __name__ == "__main__":
    raise SystemExit(main())
