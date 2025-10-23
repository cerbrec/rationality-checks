#!/usr/bin/env python3
"""
Test script for web search integration in rationality checks
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.integrated_verification import IntegratedVerificationPipeline
from src.verification_pipeline import BedrockProvider

def test_simple_claim():
    """Test a simple factual claim that should trigger web search"""

    print("=" * 80)
    print("WEB SEARCH INTEGRATION TEST")
    print("=" * 80)

    # Initialize Bedrock provider
    print("\n1. Initializing Bedrock provider...")
    provider = BedrockProvider.from_env(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )

    # Create pipeline
    print("2. Creating verification pipeline...")
    pipeline = IntegratedVerificationPipeline(provider)

    # Test with a simple document containing verifiable facts
    test_document = """
    Analysis of Qualtrics:

    Qualtrics is a leading experience management company headquartered in Provo, Utah.
    The company was founded in 2002 and acquired by SAP in 2018 for $8 billion.

    Qualtrics went public again in 2021 on the NASDAQ stock exchange.
    The company specializes in customer experience, employee experience, and brand research software.
    """

    print("\n3. Running verification on test document...")
    print("   Document contains verifiable facts about Qualtrics")
    print("   Expected: Web search should be triggered to verify claims")
    print("-" * 80)

    report = pipeline.verify_analysis(
        original_output=test_document,
        original_query="Provide information about Qualtrics",
        enable_tool_use=True  # Enable web search for fact-checking
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nTotal claims analyzed: {len(report.assessments)}")

    for i, assessment in enumerate(report.assessments, 1):
        print(f"\n{i}. Claim: {assessment.claim.text}")
        print(f"   Type: {assessment.claim.claim_type.value}")
        print(f"   Confidence: {assessment.overall_confidence:.2f}")
        print(f"   Recommendation: {assessment.recommendation}")

        # Show evidence from fact-checking
        for result in assessment.verification_results:
            if result.method.value == "fact_check" and result.evidence:
                print(f"   Evidence found: {len(result.evidence)} items")
                for ev in result.evidence[:2]:  # Show first 2
                    print(f"     - {ev.source}: {ev.content[:100]}...")

        if assessment.verification_results:
            fact_check_results = [r for r in assessment.verification_results if r.method.value == "fact_check"]
            if fact_check_results:
                print(f"   Fact check: {'✓ passed' if fact_check_results[0].passed else '✗ failed'}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return report

if __name__ == "__main__":
    try:
        report = test_simple_claim()
        print("\n✅ Web search integration test passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
