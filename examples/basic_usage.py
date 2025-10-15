"""
Basic Usage Examples for Rationality LLM Verification Pipeline

This script demonstrates how to use the integrated verification pipeline
to verify LLM-generated analysis.
"""

import os
import sys
from typing import Optional

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrated_verification import IntegratedVerificationPipeline
from verification_pipeline import AnthropicProvider, OpenAIProvider, MockLLMProvider


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_llm_provider():
    """
    Get LLM provider based on environment variables.
    Falls back to Mock provider for testing.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("Using Anthropic Claude provider")
        return AnthropicProvider(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-3-5-sonnet-20241022"
        )
    elif os.environ.get("OPENAI_API_KEY"):
        print("Using OpenAI GPT provider")
        return OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4"
        )
    else:
        print("No API key found. Using Mock provider for demonstration.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to use real LLM.")
        return MockLLMProvider()


# ============================================================================
# EXAMPLE 1: FINANCIAL ANALYSIS WITH MATHEMATICAL CONTRADICTION
# ============================================================================

def example_financial_contradiction():
    """
    Example showing how the pipeline catches mathematical inconsistencies
    in financial analysis.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Financial Analysis with Mathematical Contradiction")
    print("=" * 80)

    # Original LLM output with a mathematical error
    original_output = """
    Company X Financial Analysis:

    Based on our research, Company X is currently valued at $50 billion.
    The company reported annual revenue of $7 billion last quarter.

    Using industry-standard valuation metrics, we applied a 10x revenue
    multiple, which is typical for high-growth technology companies in
    this sector.

    The company has demonstrated strong competitive advantages through
    its proprietary technology platform and established market position.
    Management has a proven track record of execution.

    Recommendation: The current valuation appears reasonable given the
    company's growth trajectory and market position.
    """

    original_query = "Analyze Company X's valuation and provide investment recommendation"

    # Create pipeline and verify
    llm = get_llm_provider()
    pipeline = IntegratedVerificationPipeline(llm)

    print("\n📝 Original Output:")
    print("-" * 80)
    print(original_output)

    print("\n🔍 Running Verification Pipeline...")
    print("-" * 80)

    # Note: With MockLLMProvider, this won't do full verification
    # Use real API key to see actual results
    if isinstance(llm, MockLLMProvider):
        print("⚠️  Using Mock provider - showing expected behavior:")
        print("\nExpected Results:")
        print("\nClaim: 'Company X is valued at $50B'")
        print("  Type: QUANTITATIVE (formalizable)")
        print("  Method: World State Verification")
        print("  Result: ❌ FAILED")
        print("  Issue: Constraint violated: 50B ≠ 10 * 7B (50B ≠ 70B)")
        print("  Confidence: 1.0 (mathematical proof)")
        print("  Recommendation: REVISE")
        print("\nClaim: 'Has strong competitive advantages'")
        print("  Type: INTERPRETIVE (non-formalizable)")
        print("  Method: LLM Empirical Testing")
        print("  Result: ⚠️  UNCERTAIN")
        print("  Confidence: 0.7 (needs evidence)")
        print("  Recommendation: FLAG_UNCERTAINTY")
    else:
        report = pipeline.verify_analysis(original_output, original_query)

        print("\n📊 Verification Results:")
        print("-" * 80)

        for i, assessment in enumerate(report.assessments, 1):
            print(f"\n{i}. Claim: {assessment.claim.text}")
            print(f"   Type: {assessment.claim.claim_type.value}")
            print(f"   Confidence: {assessment.overall_confidence:.2f}")
            print(f"   Recommendation: {assessment.recommendation}")

            if not all(r.passed for r in assessment.verification_results):
                print(f"   Issues:")
                for result in assessment.verification_results:
                    for issue in result.issues_found:
                        print(f"     - {issue}")

        print("\n✅ Improved Output:")
        print("-" * 80)
        print(report.improved_output)


# ============================================================================
# EXAMPLE 2: RESEARCH PAPER VERIFICATION
# ============================================================================

def example_research_verification():
    """
    Example verifying research claims with mixed formalizable and
    interpretive content.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Research Paper Verification")
    print("=" * 80)

    original_output = """
    Study Results Summary:

    Our experiment included 100 participants randomly assigned to treatment
    and control groups. The treatment group (n=50) received the intervention
    while the control group (n=60) received standard care.

    We observed a 25% improvement in outcomes for the treatment group
    compared to baseline. The control group showed a 5% improvement.

    Statistical analysis revealed a p-value of 0.03, indicating statistical
    significance. The effect size (Cohen's d) was 0.65, suggesting a
    moderate to large practical effect.

    These findings suggest that the intervention is highly effective and
    should be considered for broader implementation.
    """

    original_query = "Summarize the research findings and their implications"

    llm = get_llm_provider()
    pipeline = IntegratedVerificationPipeline(llm)

    print("\n📝 Original Output:")
    print("-" * 80)
    print(original_output)

    print("\n🔍 Expected Issues:")
    print("-" * 80)
    print("1. Sample size inconsistency: n=50 + n=60 = 110, not 100")
    print("2. 'Highly effective' is strong claim based on p=0.03")
    print("3. Missing confidence intervals")
    print("4. No discussion of limitations")

    if not isinstance(llm, MockLLMProvider):
        report = pipeline.verify_analysis(original_output, original_query)
        print("\n📊 Actual Verification Results:")
        print("-" * 80)
        for assessment in report.assessments:
            if not all(r.passed for r in assessment.verification_results):
                print(f"\n⚠️  {assessment.claim.text}")
                print(f"   Confidence: {assessment.overall_confidence:.2f}")


# ============================================================================
# EXAMPLE 3: TECHNICAL SPECIFICATION VERIFICATION
# ============================================================================

def example_technical_spec():
    """
    Example verifying technical specifications with quantitative claims.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Technical Specification Verification")
    print("=" * 80)

    original_output = """
    System Performance Specifications:

    The new API can handle 10,000 requests per second with an average
    response time of 50ms. Under load testing, we observed 99.9% uptime.

    The system processes approximately 864 million requests per day,
    operating 24/7 with redundant failover systems.

    Memory usage peaks at 16GB during high load, with average utilization
    around 8GB. CPU usage averages 40% with peaks up to 95% during
    traffic spikes.

    Based on these metrics, the system can support our expected growth
    for the next 2-3 years without additional infrastructure investment.
    """

    original_query = "Document system performance and capacity"

    print("\n📝 Original Output:")
    print("-" * 80)
    print(original_output)

    print("\n🔍 Verification Check:")
    print("-" * 80)
    print("Mathematical verification:")
    print("  10,000 req/sec * 86,400 sec/day = 864,000,000 req/day ✓")
    print("  Claim consistency: PASSED")


# ============================================================================
# EXAMPLE 4: CUSTOM VERIFICATION WORKFLOW
# ============================================================================

def example_custom_workflow():
    """
    Example showing how to customize the verification workflow.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Custom Verification Workflow")
    print("=" * 80)

    llm = get_llm_provider()
    pipeline = IntegratedVerificationPipeline(llm)

    # Example: Only verify quantitative claims
    original_output = "Revenue is $10M. Profit margin is strong. Growth is 20%."
    original_query = "Summarize financials"

    print("\n📝 Demonstrating custom filtering:")
    print("-" * 80)
    print("You could extend the pipeline to:")
    print("1. Filter specific claim types")
    print("2. Apply custom confidence thresholds")
    print("3. Use different verification methods")
    print("4. Integrate with external data sources")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("RATIONALITY LLM - VERIFICATION PIPELINE EXAMPLES")
    print("=" * 80)

    print("\n⚠️  To use real LLM verification, set environment variables:")
    print("   export ANTHROPIC_API_KEY='your-key'  # For Claude")
    print("   export OPENAI_API_KEY='your-key'     # For GPT")

    # Run examples
    example_financial_contradiction()
    example_research_verification()
    example_technical_spec()
    example_custom_workflow()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
