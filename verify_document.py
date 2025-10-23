#!/usr/bin/env python3
"""
Document Verification CLI Tool

Simple command-line tool to run rationality checks on any document.

Usage:
    python verify_document.py document.md
    python verify_document.py document.txt --provider anthropic
    python verify_document.py document.md --provider bedrock --model nova-pro
    python verify_document.py document.md --output report.json
"""

import os
import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent))

from src.integrated_verification import IntegratedVerificationPipeline
from src.verification_pipeline import (
    AnthropicProvider,
    OpenAIProvider,
    BedrockProvider,
    GeminiProvider
)


def get_provider(provider_name: str, model: str = None):
    """Initialize the specified LLM provider"""

    if provider_name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ Error: ANTHROPIC_API_KEY not found in .env file")
            sys.exit(1)
        model = model or "claude-3-5-sonnet-20241022"
        return AnthropicProvider(api_key=api_key, model=model)

    elif provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Error: OPENAI_API_KEY not found in .env file")
            sys.exit(1)
        model = model or "gpt-4o"
        return OpenAIProvider(api_key=api_key, model=model)

    elif provider_name == "bedrock":
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            print("❌ Error: AWS_ACCESS_KEY_ID not found in .env file")
            sys.exit(1)
        model = model or "us.amazon.nova-pro-v1:0"
        return BedrockProvider(region_name=aws_region, model_id=model)

    elif provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Error: GEMINI_API_KEY not found in .env file")
            sys.exit(1)
        model = model or "gemini-2.0-flash-exp"
        return GeminiProvider(api_key=api_key, model=model)

    else:
        print(f"❌ Error: Unknown provider '{provider_name}'")
        print("   Available providers: anthropic, openai, bedrock, gemini")
        sys.exit(1)


def format_assessment(assessment, index):
    """Format a single claim assessment for display"""
    lines = []

    # Header
    claim_preview = assessment.claim.text[:100]
    if len(assessment.claim.text) > 100:
        claim_preview += "..."

    lines.append(f"\n{index}. 📍 {claim_preview}")
    lines.append(f"   Type: {assessment.claim.claim_type.value}")
    lines.append(f"   Confidence: {assessment.overall_confidence:.2f}")
    lines.append(f"   Recommendation: {assessment.recommendation.upper()}")

    # Show issues if any
    has_issues = not all(r.passed for r in assessment.verification_results)
    if has_issues:
        lines.append(f"   Issues:")
        for result in assessment.verification_results:
            if not result.passed:
                for issue in result.issues_found[:2]:  # Limit to 2 issues per result
                    lines.append(f"     • {issue}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Run rationality checks on a document',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_document.py analysis.md
  python verify_document.py report.txt --provider bedrock --model nova-pro
  python verify_document.py data.md --output results.json --verbose

Available providers:
  anthropic  - Claude models (default: claude-3-5-sonnet-20241022)
  openai     - GPT models (default: gpt-4o)
  bedrock    - AWS Bedrock (default: nova-pro)
  gemini     - Google Gemini (default: gemini-2.0-flash-exp)
        """
    )

    parser.add_argument('document', help='Path to document to verify')
    parser.add_argument('--provider', default='anthropic',
                       choices=['anthropic', 'openai', 'bedrock', 'gemini'],
                       help='LLM provider to use (default: anthropic)')
    parser.add_argument('--model', help='Specific model to use (optional)')
    parser.add_argument('--output', help='Save detailed report to JSON file')
    parser.add_argument('--query', help='Original query context (optional)')
    parser.add_argument('--verbose', action='store_true', help='Show all claims, not just issues')

    args = parser.parse_args()

    # Read document
    doc_path = Path(args.document).expanduser()
    if not doc_path.exists():
        print(f"❌ Error: File not found: {doc_path}")
        sys.exit(1)

    print("=" * 80)
    print("DOCUMENT VERIFICATION - RATIONALITY CHECK")
    print("=" * 80)
    print(f"\n📄 Document: {doc_path}")

    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)

    print(f"   Size: {len(document_text)} characters")

    # Initialize provider
    print(f"\n🤖 Provider: {args.provider}")
    if args.model:
        print(f"   Model: {args.model}")

    try:
        llm = get_provider(args.provider, args.model)
    except Exception as e:
        print(f"❌ Error initializing provider: {e}")
        sys.exit(1)

    # Create pipeline
    print("🔧 Initializing verification pipeline...")
    pipeline = IntegratedVerificationPipeline(llm)

    # Run verification
    query = args.query or "Analyze and verify the claims in this document"

    print("\n🔍 Running verification...")
    print("   This may take 2-3 minutes depending on document length...")
    print("   Processing steps:")
    print("   • Extracting claims")
    print("   • Classifying claim types")
    print("   • Verifying mathematical consistency")
    print("   • Checking logical contradictions")
    print("   • Assessing confidence levels")
    print("-" * 80)

    try:
        report = pipeline.verify_analysis(document_text, query)
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Categorize results
    passed = [a for a in report.assessments if a.recommendation == "keep"]
    flagged = [a for a in report.assessments if a.recommendation in ["flag_uncertainty", "flag"]]
    failed = [a for a in report.assessments if a.recommendation in ["revise", "remove"]]

    # Display summary
    print("\n" + "=" * 80)
    print("📊 VERIFICATION RESULTS")
    print("=" * 80)
    print(f"\nTotal claims analyzed: {len(report.assessments)}")
    print(f"  ✅ Passed: {len(passed)}")
    print(f"  ⚠️  Flagged: {len(flagged)}")
    print(f"  ❌ Failed: {len(failed)}")

    accuracy_rate = (len(passed) / len(report.assessments) * 100) if report.assessments else 0
    print(f"\nAccuracy Rate: {accuracy_rate:.1f}%")

    # Show failed claims
    if failed:
        print("\n" + "=" * 80)
        print("❌ FAILED CLAIMS (Require Revision)")
        print("=" * 80)
        for i, assessment in enumerate(failed, 1):
            print(format_assessment(assessment, i))

    # Show flagged claims
    if flagged:
        print("\n" + "=" * 80)
        print("⚠️  FLAGGED CLAIMS (Need Additional Evidence)")
        print("=" * 80)
        for i, assessment in enumerate(flagged, 1):
            print(format_assessment(assessment, i))

    # Show passed claims if verbose
    if args.verbose and passed:
        print("\n" + "=" * 80)
        print("✅ PASSED CLAIMS")
        print("=" * 80)
        for i, assessment in enumerate(passed[:20], 1):  # Limit to first 20
            print(f"\n{i}. {assessment.claim.text[:80]}...")
            print(f"   Confidence: {assessment.overall_confidence:.2f}")
        if len(passed) > 20:
            print(f"\n   ... and {len(passed) - 20} more")

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "document": str(doc_path),
            "provider": args.provider,
            "model": args.model,
            "summary": {
                "total_claims": len(report.assessments),
                "passed": len(passed),
                "flagged": len(flagged),
                "failed": len(failed),
                "accuracy_rate": accuracy_rate
            },
            "assessments": [
                {
                    "claim": a.claim.text,
                    "type": a.claim.claim_type.value,
                    "confidence": a.overall_confidence,
                    "recommendation": a.recommendation,
                    "issues": [
                        issue
                        for result in a.verification_results
                        for issue in result.issues_found
                    ]
                }
                for a in report.assessments
            ],
            "improved_output": report.improved_output
        }

        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Detailed report saved to: {output_path}")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not save report: {e}")

    # Final recommendations
    print("\n" + "=" * 80)
    print("📝 RECOMMENDATIONS")
    print("=" * 80)

    if failed:
        print("\n⚠️  CRITICAL: Review and revise failed claims before publishing")
        print("   These claims have logical contradictions or unsupported assertions")
    elif flagged:
        print("\n⚠️  SUGGESTED: Add supporting evidence for flagged claims")
        print("   These claims need additional context or verification")
    else:
        print("\n✅ Document appears sound and well-supported")
        print("   All claims passed verification checks")

    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)

    # Exit with appropriate code
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
