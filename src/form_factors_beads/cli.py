"""Command-line entry point."""

from .core import main as _main


def main() -> int:
    """Run the form-factor command-line interface."""
    return _main()
