"""
Athlete Report Verification Test

Specific test for the FLX athlete report with expected claims
and interesting test scenarios.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.test_interactive_verification import InteractiveVerificationTest
from src.verification_pipeline import AnthropicProvider, OpenAIProvider, BedrockProvider, MockLLMProvider


# ============================================================================
# EXPECTED CLAIMS CATALOG
# ============================================================================

EXPECTED_CLAIMS = {
    "quantitative": [
        "Height: 6'0\" (72 inches)",
        "Weight: 190 lbs",
        "40-yard dash: 4.72 seconds",
        "Shuttle: 4.31 seconds",
        "Vertical: 31 inches",
        "Bench Press: 235 lbs",
        "Protein: 165-190g per day",
        "Carbohydrates: 225-275g per day",
        "Fats: 70-85g per day",
        "Hydration: 80-100oz per day",
    ],
    "factual": [
        "School: Corner Canyon (Utah)",
        "Class: 2025",
        "Position: QB - Dual Threat",
        "Comparable to Jayden Daniels",
        "Comparable to Marcel Reed (6'1\" 185 lbs)",
    ],
    "interpretive": [
        "Silent Assassin profile",
        "Elite spatial awareness",
        "Among rare <2% of high school players",
        "Sunday-level ceiling",
        "Strong competitive moat",
    ],
}

INTERESTING_SCENARIOS = {
    "protein_math": {
        "description": "Protein distribution across meals",
        "claims": [
            "Total protein: 165-190g/day",
            "Per meal: 30-45g protein",
            "Meals: 3+ meals per day",
        ],
        "constraint": "If 4 meals/day: 30-45g × 4 = 120-180g (consistent with 165-190g)",
        "expected": "CONSISTENT",
    },
    "comparable_builds": {
        "description": "Physical comparisons to other QBs",
        "claims": [
            "Helaman: 6'0\" 190 lbs",
            "Marcel Reed: 6'1\" 185 lbs (nearly identical)",
            "Jayden Daniels: slightly taller (similar ability)",
        ],
        "expected": "Need to check if comparisons are reasonable",
    },
    "performance_percentile": {
        "description": "Elite status claim",
        "claims": [
            "Among rare <2% of high school players",
            "4.72 sec 40-yard dash",
            "31\" vertical jump",
        ],
        "question": "Do these measurables actually put him in top 2%?",
        "expected": "Needs fact-checking",
    },
}


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_with_flx_report(llm_provider, report_path: str, auto_continue: bool = False):
    """
    Run interactive test with the FLX athlete report.

    Args:
        llm_provider: LLM provider to use
        report_path: Path to the FLX report JSON
        auto_continue: If True, don't pause between steps
    """
    print("\n" + "=" * 80)
    print("ATHLETE REPORT VERIFICATION TEST")
    print("=" * 80)
    print(f"\nReport: {report_path}")
    print(f"Provider: {llm_provider.__class__.__name__}")
    print("\n" + "=" * 80)

    # Create test instance
    test = InteractiveVerificationTest(llm_provider, auto_continue=auto_continue)

    # Run test
    try:
        test.run_from_json(
            report_path,
            query="Analyze athlete performance report for accuracy and consistency"
        )
        print("\n✓ Test completed successfully!")

    except FileNotFoundError:
        print(f"\n❌ Error: Report file not found at {report_path}")
        print("Please check the file path and try again.")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_expected_findings():
    """Print what we expect to find in the report"""
    print("\n" + "=" * 80)
    print("EXPECTED FINDINGS")
    print("=" * 80)

    print("\nQuantitative Claims (should be formalizable):")
    for claim in EXPECTED_CLAIMS["quantitative"]:
        print(f"  • {claim}")

    print("\nFactual Claims (should be formalizable):")
    for claim in EXPECTED_CLAIMS["factual"]:
        print(f"  • {claim}")

    print("\nInterpretive Claims (not formalizable):")
    for claim in EXPECTED_CLAIMS["interpretive"]:
        print(f"  • {claim}")

    print("\n" + "=" * 80)
    print("INTERESTING TEST SCENARIOS")
    print("=" * 80)

    for scenario_name, scenario in INTERESTING_SCENARIOS.items():
        print(f"\n{scenario_name.upper().replace('_', ' ')}:")
        print(f"Description: {scenario['description']}")
        print("Claims:")
        for claim in scenario["claims"]:
            print(f"  • {claim}")
        if "constraint" in scenario:
            print(f"Constraint: {scenario['constraint']}")
        print(f"Expected: {scenario['expected']}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Test athlete report verification")
    parser.add_argument(
        "--report",
        default="/Users/drw/cerbrec/code-conversion/resources/flx-resources/flx-report-payload.json",
        help="Path to athlete report JSON file"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "bedrock", "mock"],
        default="mock",
        help="LLM provider to use"
    )
    parser.add_argument(
        "--bedrock-model",
        default="us.anthropic.claude-sonnet-4-20250514-v1:0",
        help="Bedrock model ID (only used with --provider=bedrock)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-continue without pauses (for automated testing)"
    )
    parser.add_argument(
        "--expected",
        action="store_true",
        help="Show expected findings and exit"
    )

    args = parser.parse_args()

    if args.expected:
        print_expected_findings()
        return

    # Initialize LLM provider
    if args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)
        llm = AnthropicProvider(api_key=api_key)

    elif args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ Error: OPENAI_API_KEY environment variable not set")
            sys.exit(1)
        llm = OpenAIProvider(api_key=api_key)

    elif args.provider == "bedrock":
        # Check for AWS credentials
        if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            print("❌ Error: AWS credentials not set")
            print("   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            print("   Optionally set AWS_REGION (defaults to us-east-1)")
            sys.exit(1)
        print(f"Using AWS Bedrock with model: {args.bedrock_model}")
        llm = BedrockProvider.from_env(model_id=args.bedrock_model)

    else:  # mock
        print("⚠️  Using Mock LLM provider - no actual verification will occur")
        print("   Set --provider=anthropic, --provider=openai, or --provider=bedrock for real testing")
        llm = MockLLMProvider()

    # Run test
    test_with_flx_report(llm, args.report, auto_continue=args.auto)


if __name__ == "__main__":
    main()
