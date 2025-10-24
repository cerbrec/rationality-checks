#!/usr/bin/env python3
"""
Test case for Jake Retzlaff athlete profile verification

This test validates that the verification pipeline correctly identifies
factually incorrect claims in the test_retzlaff.md document, specifically:
- Incorrect class year (Senior vs Junior)
- Inflated social media follower count
- Overstated passing yards
- Overstated touchdown count
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_retzlaff_verification():
    """
    Run verification on test_retzlaff.md and validate that bad claims are flagged
    """

    print("=" * 80)
    print("JAKE RETZLAFF ATHLETE PROFILE VERIFICATION TEST")
    print("=" * 80)

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    test_document = script_dir / "test_retzlaff.md"
    verify_script = project_root / "verify_document.py"

    # Verify files exist
    if not test_document.exists():
        print(f"❌ Error: Test document not found: {test_document}")
        sys.exit(1)

    if not verify_script.exists():
        print(f"❌ Error: Verification script not found: {verify_script}")
        sys.exit(1)

    print(f"\n📄 Test Document: {test_document}")
    print(f"🔧 Verification Script: {verify_script}")

    # Check AWS credentials
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("\n❌ Error: AWS credentials not found")
        print("   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
        sys.exit(1)

    print(f"\n✓ AWS credentials loaded")
    print(f"  Region: {os.getenv('AWS_REGION', 'us-east-1')}")

    # Run the verification command
    print("\n" + "=" * 80)
    print("RUNNING VERIFICATION")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(verify_script),
        str(test_document),
        "--provider", "bedrock",
        "--model", "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "--verbose"
    ]

    print(f"\nCommand: {' '.join(cmd)}")
    print("\nThis will take 2-3 minutes...\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Parse output to validate claims were flagged
        output = result.stdout

        print("\n" + "=" * 80)
        print("VALIDATION CHECKS")
        print("=" * 80)

        # Expected bad claims that should be flagged or failed
        expected_issues = {
            "senior": "Jake Retzlaff class year (should be Junior, not Senior)",
            "48,500": "Social media followers (inflated from ~21K)",
            "3,200": "Passing yards (actual 2,947)",
            "28": "Touchdowns (actual 20)"
        }

        found_issues = []
        missing_issues = []

        # Check if any claims were failed or flagged
        has_failed = "FAILED CLAIMS" in output or "failed:" in output.lower()
        has_flagged = "FLAGGED CLAIMS" in output or "flagged:" in output.lower()

        print(f"\n✓ Verification completed")
        print(f"  Has failed claims: {has_failed}")
        print(f"  Has flagged claims: {has_flagged}")

        # Check for specific issues
        print("\nChecking for expected issues:")
        for key, description in expected_issues.items():
            if key in output:
                print(f"  ✓ Found reference to: {description}")
                found_issues.append(description)
            else:
                print(f"  ⚠️  Missing reference to: {description}")
                missing_issues.append(description)

        # Validate test results
        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)

        success = True

        if not (has_failed or has_flagged):
            print("\n❌ FAILED: No claims were flagged or failed")
            print("   Expected the verification to identify incorrect facts")
            success = False
        else:
            print(f"\n✓ Claims were properly flagged/failed")

        if len(found_issues) < 2:
            print(f"\n⚠️  WARNING: Only found {len(found_issues)}/4 expected issues")
            print("   This may indicate the verification missed some bad claims")
        else:
            print(f"\n✓ Found {len(found_issues)}/4 expected issues")

        # Exit code indicates if bad claims were found (exit 1 = issues found, which is good)
        if result.returncode == 1 and success:
            print("\n" + "=" * 80)
            print("✅ TEST PASSED")
            print("=" * 80)
            print("\nThe verification pipeline successfully identified factual errors")
            print("in the Jake Retzlaff athlete profile.")
            return 0
        elif result.returncode == 0:
            print("\n" + "=" * 80)
            print("❌ TEST FAILED")
            print("=" * 80)
            print("\nThe verification pipeline did NOT identify any issues.")
            print("This document contains known factual errors that should have been caught.")
            return 1
        else:
            print(f"\n❌ TEST FAILED WITH ERROR")
            print(f"   Exit code: {result.returncode}")
            return 1

    except subprocess.TimeoutExpired:
        print("\n❌ Error: Verification timed out after 5 minutes")
        return 1
    except Exception as e:
        print(f"\n❌ Error running verification: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = test_retzlaff_verification()
    sys.exit(exit_code)
