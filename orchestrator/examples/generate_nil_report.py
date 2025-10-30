#!/usr/bin/env python3
"""
NIL Report Generator
====================

Generate intelligent NIL (Name, Image, Likeness) college player valuation reports
with integrated rationality verification.

Usage:
    python generate_nil_report.py --player "Travis Hunter" --school "Colorado" --position "WR/DB"
    python generate_nil_report.py --player "Shedeur Sanders" --school "Colorado" --position "QB" --output sanders_report.json
    python generate_nil_report.py --query "Evaluate top QB prospects for NIL deals"

Features:
- 7-step intelligent report generation
- Integrated verification (stats, valuations, predictions)
- Player stats verification against NCAA data
- NIL market comparisons
- Mathematical consistency checks
- Confidence scores for all claims

Example Output:
- Player performance analysis (verified stats)
- NIL market valuation (with comparable players)
- Future value prediction (adversarially reviewed)
- Verification summary (confidence scores)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.unified_orchestrator import IntelligentOrchestrator


def format_report(report):
    """Format report for console display"""
    lines = []

    lines.append("\n" + "=" * 80)
    lines.append("NIL PLAYER VALUATION REPORT")
    lines.append("=" * 80)

    # Goal
    lines.append(f"\n📋 OBJECTIVE:")
    lines.append(f"   {report.goal.get('objective', 'N/A')}")

    # Strategy
    lines.append(f"\n🎯 APPROACH:")
    lines.append(f"   {report.strategy.get('approach', 'N/A')[:200]}...")

    # Key Data
    if report.collected_data.get('collected_data'):
        lines.append(f"\n📊 DATA COLLECTED:")
        for key, value in list(report.collected_data.get('collected_data', {}).items())[:3]:
            lines.append(f"   • {key}: {value}")

    # Prediction
    lines.append(f"\n🔮 PREDICTION:")
    lines.append(f"   Outcome: {report.prediction.get('predicted_outcome', 'N/A')}")
    lines.append(f"   Success Probability: {report.prediction.get('success_probability', 0.0):.1%}")
    lines.append(f"   Confidence: {report.prediction.get('confidence_score', 0.0):.2f}")

    # Final Recommendation
    lines.append(f"\n✅ RECOMMENDATION:")
    lines.append(f"   {report.final_recommendation}")

    # Implementation Plan
    if report.implementation_plan:
        lines.append(f"\n📝 IMPLEMENTATION PLAN:")
        for i, step in enumerate(report.implementation_plan, 1):
            lines.append(f"   {i}. {step}")

    # Verification Summary
    lines.append(f"\n" + "=" * 80)
    lines.append("🔍 VERIFICATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"   Total Verifications: {report.verification_summary['total_verifications']}")
    lines.append(f"   Passed: {report.verification_summary['passed']}")
    lines.append(f"   Failed: {report.verification_summary['failed']}")
    lines.append(f"   Overall Confidence: {report.overall_confidence:.2f}")
    lines.append(f"   Issues Found: {report.verification_summary['issues_found']}")

    # Show verification details if any failed
    failed_verifications = [v for v in report.verified_claims if not v.passed]
    if failed_verifications:
        lines.append(f"\n⚠️  VERIFICATION ISSUES:")
        for v in failed_verifications[:3]:  # Show first 3
            lines.append(f"   • {v.claim}")
            lines.append(f"     Step: {v.step}, Confidence: {v.confidence:.2f}")
            if v.issues:
                for issue in v.issues[:2]:
                    lines.append(f"     - {issue}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def save_report_json(report, output_path):
    """Save report to JSON file"""
    # Convert dataclasses to dicts for JSON serialization
    report_dict = {
        "domain": report.domain,
        "goal": report.goal,
        "strategy": report.strategy,
        "collected_data": report.collected_data,
        "processed_data": report.processed_data,
        "connections": report.connections,
        "prediction": report.prediction,
        "final_recommendation": report.final_recommendation,
        "implementation_plan": report.implementation_plan,
        "verification_summary": report.verification_summary,
        "overall_confidence": report.overall_confidence,
        "verified_claims": [
            {
                "claim": v.claim,
                "step": v.step,
                "verification_methods": v.verification_methods,
                "confidence": v.confidence,
                "passed": v.passed,
                "issues": v.issues
            }
            for v in report.verified_claims
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Generate NIL player valuation report with integrated verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_nil_report.py --player "Travis Hunter" --school "Colorado" --position "WR/DB"
  python generate_nil_report.py --query "Analyze top QB prospects for NIL value"
  python generate_nil_report.py --player "Player Name" --school "School" --output report.json --verbose

Features:
  ✓ 7-step intelligent report generation
  ✓ Integrated verification (stats, valuations, predictions)
  ✓ Mathematical consistency checks
  ✓ Confidence scores for all claims
  ✓ Verification summary with issues found

Note:
  Requires ANTHROPIC_API_KEY environment variable.
  Set it in .env file or export ANTHROPIC_API_KEY="your-key-here"
        """
    )

    parser.add_argument('--player', help='Player name')
    parser.add_argument('--school', help='College/university name')
    parser.add_argument('--position', help='Player position (QB, WR, RB, etc.)')
    parser.add_argument('--season', default='2024', help='Season year (default: 2024)')

    parser.add_argument('--query', help='Custom query instead of player-specific')
    parser.add_argument('--context', default='', help='Additional context')

    parser.add_argument('--output', help='Save report to JSON file')
    parser.add_argument('--model', default='claude-sonnet-4-5-20250929',
                       help='Claude model to use (default: sonnet 4.5)')
    parser.add_argument('--no-verification', action='store_true',
                       help='Disable integrated verification')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress')

    args = parser.parse_args()

    # Build query
    if args.query:
        query = args.query
        context = args.context
    elif args.player and args.school:
        query = f"Evaluate {args.player}'s NIL market value"
        context_parts = [
            f"Player: {args.player}",
            f"School: {args.school}",
        ]
        if args.position:
            context_parts.append(f"Position: {args.position}")
        if args.season:
            context_parts.append(f"Season: {args.season}")
        context = ", ".join(context_parts)
    else:
        print("❌ Error: Must provide either --query or (--player + --school)")
        parser.print_help()
        sys.exit(1)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found")
        print("   Set it in .env file or: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    print("=" * 80)
    print("NIL REPORT GENERATOR")
    print("=" * 80)
    print(f"\n🎯 Query: {query}")
    print(f"📝 Context: {context}")
    print(f"🤖 Model: {args.model}")
    print(f"🔍 Verification: {'Enabled' if not args.no_verification else 'Disabled'}")

    try:
        # Initialize orchestrator
        orchestrator = IntelligentOrchestrator(
            domain="nil",
            model=args.model,
            enable_verification=not args.no_verification
        )

        # Generate report
        print(f"\n⏳ Generating report...")
        report = orchestrator.generate_report(query=query, context=context)

        # Display report
        print(format_report(report))

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            save_report_json(report, output_path)
            print(f"\n💾 Report saved to: {output_path}")

        # Exit code based on verification
        if report.verification_summary['failed'] > 0:
            print("\n⚠️  Warning: Some verification checks failed")
            print("   Review the report for details")
            sys.exit(1)
        else:
            print("\n✅ Report generation complete")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
