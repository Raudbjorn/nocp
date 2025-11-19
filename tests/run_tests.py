#!/usr/bin/env python3
"""
NOCP Test Runner

Comprehensive test runner with category organization, progress tracking,
and detailed reporting.

Categories:
- unit: Fast, isolated unit tests
- integration: Component integration tests
- e2e: End-to-end workflow tests
- performance: Performance benchmarks

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --category unit    # Run only unit tests
    python tests/run_tests.py -v                 # Verbose output
    python tests/run_tests.py --fail-fast        # Stop on first failure
    python tests/run_tests.py --json report.json # Generate JSON report
"""

import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class TestCategory(Enum):
    """Test categories"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"

@dataclass
class TestResult:
    """Test result for a category"""
    category: str
    passed: bool
    duration_s: float
    test_count: int
    passed_count: int
    failed_count: int
    output: str

@dataclass
class TestSummary:
    """Overall test summary"""
    total_tests: int
    total_passed: int
    total_failed: int
    total_duration_s: float
    success_rate: float
    results: List[TestResult]

class NOCPTestRunner:
    """Test runner for NOCP project"""

    def __init__(
        self,
        verbose: bool = False,
        fail_fast: bool = False,
        capture_output: bool = True
    ):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = Path(__file__).parent
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.capture_output = capture_output
        self.results: List[TestResult] = []

    def discover_tests(
        self,
        category: Optional[TestCategory] = None
    ) -> Dict[TestCategory, List[Path]]:
        """Discover all test files organized by category"""
        categories = {
            TestCategory.UNIT: self.tests_dir / "unit",
            TestCategory.INTEGRATION: self.tests_dir / "integration",
            TestCategory.E2E: self.tests_dir / "e2e",
            TestCategory.PERFORMANCE: self.tests_dir / "performance",
        }

        discovered = {}

        for cat_enum, cat_path in categories.items():
            # Filter by category if specified
            if category and category != cat_enum:
                continue

            # Skip if directory doesn't exist
            if not cat_path.exists():
                continue

            # Find all test_*.py files
            test_files = list(cat_path.rglob("test_*.py"))
            if test_files:
                discovered[cat_enum] = sorted(test_files)

        return discovered

    def run_pytest(
        self,
        test_paths: List[Path],
        category: TestCategory
    ) -> TestResult:
        """Run pytest on given test files"""
        start_time = time.time()

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]

        # Add options
        if self.verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        if self.fail_fast:
            cmd.append("-x")

        # Add coverage for unit tests
        if category == TestCategory.UNIT:
            cmd.extend(["--cov=nocp", "--cov-report=term-missing"])

        # Add test paths
        cmd.extend([str(p) for p in test_paths])

        # Run pytest
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=self.capture_output,
            text=True
        )

        duration = time.time() - start_time

        # Parse output for test counts
        output = result.stdout + result.stderr
        passed_count, failed_count = self._parse_test_counts(output)
        test_count = passed_count + failed_count

        return TestResult(
            category=category.value,
            passed=result.returncode == 0,
            duration_s=duration,
            test_count=test_count,
            passed_count=passed_count,
            failed_count=failed_count,
            output=output
        )

    def _parse_test_counts(self, output: str) -> Tuple[int, int]:
        """Parse test counts from pytest output"""
        # Look for "X passed" or "X failed" in output
        import re

        # Pattern: "5 passed" or "3 failed"
        # pytest output format: "18 failed, 63 passed, 7 warnings in 16.99s"
        passed_match = re.search(r'(\d+)\s+passed', output)
        failed_match = re.search(r'(\d+)\s+failed', output)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0

        return passed, failed

    def print_banner(self):
        """Print test runner banner"""
        print("=" * 70)
        print("🚀 NOCP Test Suite")
        print("=" * 70)
        print()

    def print_category_header(self, category: TestCategory, file_count: int):
        """Print category header"""
        print(f"\n{'='*20} {category.value.upper()} TESTS {'='*20}")
        print(f"Found {file_count} test file(s)")

    def print_result(self, result: TestResult):
        """Print test result with emoji"""
        emoji = "✅" if result.passed else "❌"
        status = "PASSED" if result.passed else "FAILED"

        print(f"\n{emoji} {result.category} tests {status}")
        print(f"   Tests: {result.test_count}")
        print(f"   Duration: {result.duration_s:.2f}s")

        if not result.passed and not self.verbose:
            print(f"\n{result.output}")

    def run_all_tests(
        self,
        category: Optional[TestCategory] = None
    ) -> TestSummary:
        """Run all discovered tests"""
        self.print_banner()

        discovered = self.discover_tests(category)

        if not discovered:
            print("❌ No tests found!")
            return TestSummary(
                total_tests=0,
                total_passed=0,
                total_failed=0,
                total_duration_s=0.0,
                success_rate=0.0,
                results=[]
            )

        categories_list = [cat.value for cat in discovered.keys()]
        print(f"📋 Test categories: {', '.join(categories_list)}\n")

        overall_success = True
        total_start = time.time()

        for cat_enum, test_files in discovered.items():
            self.print_category_header(cat_enum, len(test_files))

            result = self.run_pytest(test_files, cat_enum)
            self.results.append(result)

            self.print_result(result)

            if not result.passed:
                overall_success = False
                if self.fail_fast:
                    break

        total_duration = time.time() - total_start

        # Generate summary
        summary = self._generate_summary(total_duration)
        self._print_summary(summary)

        return summary

    def _generate_summary(self, total_duration: float) -> TestSummary:
        """Generate test summary"""
        total_tests = sum(r.test_count for r in self.results)
        total_passed = sum(r.passed_count for r in self.results)
        total_failed = sum(r.failed_count for r in self.results)

        success_rate = (
            (total_passed / total_tests * 100)
            if total_tests > 0
            else 0.0
        )

        return TestSummary(
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            total_duration_s=total_duration,
            success_rate=success_rate,
            results=self.results
        )

    def _print_summary(self, summary: TestSummary):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)

        print(f"\nTotal Tests:    {summary.total_tests}")
        print(f"Passed:         {summary.total_passed} ✅")
        print(f"Failed:         {summary.total_failed} ❌")
        print(f"Success Rate:   {summary.success_rate:.1f}%")
        print(f"Total Duration: {summary.total_duration_s:.2f}s")

        # Category breakdown
        print("\n📋 By Category:")
        for result in summary.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {result.category:12s} {status:8s} "
                  f"({result.test_count} tests, {result.duration_s:.2f}s)")

        print("\n" + "=" * 70)

    def save_json_report(self, output_path: Path, summary: TestSummary):
        """Save JSON test report"""
        report = asdict(summary)

        with output_path.open('w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 JSON report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="NOCP test runner with category organization"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["unit", "integration", "e2e", "performance"],
        help="Run tests from specific category only"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "-x", "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Save JSON report to file"
    )

    args = parser.parse_args()

    # Convert category string to enum
    category = None
    if args.category:
        category = TestCategory(args.category)

    # Run tests
    runner = NOCPTestRunner(
        verbose=args.verbose,
        fail_fast=args.fail_fast
    )

    summary = runner.run_all_tests(category)

    # Save JSON report if requested
    if args.json:
        runner.save_json_report(args.json, summary)

    # Exit with appropriate code
    sys.exit(0 if summary.total_failed == 0 else 1)

if __name__ == "__main__":
    main()
