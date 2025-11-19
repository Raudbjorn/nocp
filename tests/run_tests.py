#!/usr/bin/env python3
"""
Custom test runner for nocp.

Provides convenient commands to run tests by category:
- Unit tests (fast, isolated)
- Integration tests (component integration)
- E2E tests (full workflows)
- Performance benchmarks

Usage:
    python tests/run_tests.py unit           # Run only unit tests
    python tests/run_tests.py integration    # Run only integration tests
    python tests/run_tests.py e2e            # Run only e2e tests
    python tests/run_tests.py performance    # Run only performance tests
    python tests/run_tests.py all            # Run all tests
    python tests/run_tests.py fast           # Run only fast tests (default)
"""

import sys
import subprocess
from pathlib import Path


def run_pytest(args: list[str]) -> int:
    """Run pytest with given arguments."""
    cmd = ["pytest"] + args
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main test runner entry point."""
    if len(sys.argv) < 2:
        category = "fast"
    else:
        category = sys.argv[1].lower()

    # Base pytest args
    base_args = ["-v", "--tb=short"]

    # Category-specific test commands
    commands = {
        "unit": ["tests/unit/", "-m", "unit or not (integration or e2e or performance)"],
        "integration": ["tests/integration/", "-m", "integration"],
        "e2e": ["tests/e2e/", "-m", "e2e", "--run-slow"],
        "performance": ["tests/performance/", "-m", "performance", "--run-slow"],
        "fast": ["tests/", "-m", "not slow"],
        "all": ["tests/", "--run-slow"],
        "slow": ["tests/", "-m", "slow", "--run-slow"],
    }

    if category not in commands:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(commands.keys())}")
        return 1

    args = base_args + commands[category]
    return run_pytest(args)


if __name__ == "__main__":
    sys.exit(main())
