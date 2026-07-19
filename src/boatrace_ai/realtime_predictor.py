"""Compatibility entry point for the realtime predictor."""

from .runtime.predictor import main


if __name__ == "__main__":
    raise SystemExit(main())
