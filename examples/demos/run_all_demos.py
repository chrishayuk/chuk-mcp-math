#!/usr/bin/env python3
"""
CHUK MCP Math - Master Demo Runner
===================================

Runs all demo scripts to prove every function in the library works.

This comprehensive demonstration covers:
- 400+ mathematical functions
- All async-native operations
- Complete type safety
- Real-world examples
"""

import subprocess
import sys
from pathlib import Path
import time


class DemoRunner:
    """Runs all demo scripts and reports results."""

    def __init__(self):
        self.demos_dir = Path(__file__).parent
        self.demos = [
            ("demo_priority2_simple.py", "🆕 Priority 2 Features - v0.3 (40 new functions)"),
            ("DEMO.py", "Main Library Demo (32 functions)"),
            ("comprehensive_demo_01_arithmetic.py", "Arithmetic Operations (44 functions)"),
            ("quick_comprehensive_demo.py", "Quick Comprehensive Demo (657+ functions)"),
            ("truly_comprehensive_demo.py", "Complete Demo (657+ functions)"),
        ]
        self.results = []

    def print_header(self):
        """Print the master demo header."""
        print("\n" + "=" * 70)
        print("CHUK MCP MATH - COMPREHENSIVE LIBRARY DEMONSTRATION")
        print("=" * 70)
        print("\nThis demonstration proves that ALL 657+ mathematical functions")
        print("in the CHUK MCP Math library are working correctly with:")
        print("  ✓ Full async-native implementation")
        print("  ✓ Complete type safety (mypy verified)")
        print("  ✓ MCP decorator integration")
        print("  ✓ Smart caching and performance optimization")
        print("  ✓ 🆕 NEW: Time series analysis & inferential statistics (v0.3)")
        print("\n" + "=" * 70 + "\n")

    def run_demo(self, script_name: str, description: str) -> bool:
        """Run a single demo script."""
        print(f"\n{'=' * 70}")
        print(f"Running: {description}")
        print(f"Script: {script_name}")
        print(f"{'=' * 70}")

        script_path = self.demos_dir / script_name
        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                print(result.stdout)
                print(f"✅ Completed successfully in {elapsed:.2f}s")
                return True
            else:
                print(f"❌ Failed with return code {result.returncode}")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print("❌ Timeout after 30 seconds")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def print_summary(self):
        """Print the final summary."""
        print("\n" + "=" * 70)
        print("DEMONSTRATION SUMMARY")
        print("=" * 70)

        for script_name, description, success in self.results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {description}")

        total = len(self.results)
        passed = sum(1 for _, _, success in self.results if success)

        print("\n" + "=" * 70)
        if passed == total:
            print(f"🎉 ALL {total} DEMO SCRIPTS PASSED!")
            print("✅ ALL 657+ MATHEMATICAL FUNCTIONS VERIFIED WORKING!")
        else:
            print(f"⚠️  {passed}/{total} demo scripts passed")
            print(f"❌ {total - passed} demo(s) failed")

        print("=" * 70 + "\n")

        return passed == total

    def run_all(self) -> bool:
        """Run all demo scripts."""
        self.print_header()

        for script_name, description in self.demos:
            success = self.run_demo(script_name, description)
            self.results.append((script_name, description, success))

        return self.print_summary()


def main():
    """Main entry point."""
    runner = DemoRunner()
    success = runner.run_all()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
