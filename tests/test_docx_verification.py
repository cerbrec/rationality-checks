#!/usr/bin/env python3
"""
Test case for DOCX document verification

This test validates that the verification pipeline correctly processes
DOCX files via Bedrock's document attachment feature and identifies
factually incorrect claims in the test_sarah_johnson.docx document.

Expected issues:
- Inflated statistics (points, assists, shooting percentage)
- Unverifiable GPA claim
- Inflated social media follower counts
- Potentially inflated recruiting ranking
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_docx_verification():
    """
    Run verification on test_sarah_johnson.docx and validate that bad claims are flagged
    """

    print("=" * 80)
    print("DOCX DOCUMENT VERIFICATION TEST")
    print("=" * 80)

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    test_document = script_dir / "test_sarah_johnson.docx"
    verify_script = project_root / "verify_document.py"

    # Create test document if it doesn't exist
    if not test_document.exists():
        print(f"\n⚠️  Test document not found, creating it...")
        create_script = script_dir / "create_test_docx.py"
        if create_script.exists():
            subprocess.run([sys.executable, str(create_script)], cwd=str(script_dir))
        else:
            print(f"❌ Error: Cannot create test document - create script not found")
            sys.exit(1)

    # Verify files exist
    if not test_document.exists():
        print(f"❌ Error: Test document not found: {test_document}")
        sys.exit(1)

    if not verify_script.exists():
        print(f"❌ Error: Verification script not found: {verify_script}")
        sys.exit(1)

    print(f"\n📄 Test Document: {test_document}")
    print(f"   Format: DOCX")
    print(f"   Size: {test_document.stat().st_size} bytes")
    print(f"🔧 Verification Script: {verify_script}")

    # Check AWS credentials
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("\n❌ Error: AWS credentials not found")
        print("   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
        print("   DOCX files require Bedrock provider with document attachment support")
        sys.exit(1)

    print(f"\n✓ AWS credentials loaded")
    print(f"  Region: {os.getenv('AWS_REGION', 'us-east-1')}")

    # Run the verification command
    print("\n" + "=" * 80)
    print("RUNNING DOCX VERIFICATION")
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
    print("\nThis will take 2-3 minutes...")
    print("Testing Bedrock document attachment feature with DOCX format\n")

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

        # Expected issues that should be flagged or failed
        expected_issues = {
            "28.5": "Points per game (unrealistically high)",
            "52%": "Three-point percentage (unrealistically high)",
            "8.2": "Assists per game (unrealistically high)",
            "4.0": "GPA claim (difficult to verify)",
            "95,000": "Instagram followers (likely inflated)",
            "42,000": "Twitter followers (likely inflated)",
            "180,000": "TikTok followers (likely inflated)",
            "five-star": "Recruiting ranking (potentially inflated)"
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
            if key.lower() in output.lower():
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

        if len(found_issues) < 3:
            print(f"\n⚠️  WARNING: Only found {len(found_issues)}/{len(expected_issues)} expected issues")
            print("   This may indicate the verification missed some questionable claims")
        else:
            print(f"\n✓ Found {len(found_issues)}/{len(expected_issues)} expected issues")

        # Exit code indicates if bad claims were found (exit 1 = issues found, which is good)
        if result.returncode == 1 and success:
            print("\n" + "=" * 80)
            print("✅ TEST PASSED")
            print("=" * 80)
            print("\nThe verification pipeline successfully processed the DOCX file")
            print("via Bedrock document attachment and identified questionable claims.")
            return 0
        elif result.returncode == 0:
            print("\n" + "=" * 80)
            print("❌ TEST FAILED")
            print("=" * 80)
            print("\nThe verification pipeline did NOT identify any issues.")
            print("This document contains known questionable claims that should have been caught.")
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
    exit_code = test_docx_verification()
    sys.exit(exit_code)
