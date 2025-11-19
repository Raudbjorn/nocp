"""
Main entry point for the nocp CLI.

This module is executed when running `python -m nocp` or via the `nocp` executable.
"""

from .cli import app


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
