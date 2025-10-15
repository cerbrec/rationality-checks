#!/usr/bin/env python3
"""
Quick start script for running LLM comparison.

Usage:
    python run_comparison.py --limit 10        # Test on 10 examples
    python run_comparison.py --limit 50        # Medium test
    python run_comparison.py --limit 100       # Full evaluation
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from verification_pipeline import OpenAIProvider, AnthropicProvider, GeminiProvider
from evaluation.runners.model_comparison import ModelComparison


def main():
    parser = argparse.ArgumentParser(description='Compare OpenAI and Claude on contradiction detection')
    parser.add_argument('--limit', type=int, default=20, help='Number of examples to test (default: 20)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress')
    parser.add_argument('--models', nargs='+', default=['gpt4', 'claude'],
                       help='Models to test: gpt4, gpt4o, gpt4-turbo, gpt5, claude, claude-opus, claude-haiku, gemini, gemini-pro')

    args = parser.parse_args()

    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    print("\n" + "=" * 80)
    print("LLM COMPARISON: Contradiction Detection Benchmark")
    print("=" * 80)

    # Check if at least one API key is available for requested models
    needs_openai = any('gpt' in m for m in args.models)
    needs_anthropic = any('claude' in m for m in args.models)
    needs_gemini = any('gemini' in m for m in args.models)

    has_valid_model = False
    if needs_openai and openai_key:
        has_valid_model = True
        print(f"\n✅ OpenAI API key loaded")
    elif needs_openai and not openai_key:
        print(f"\n⚠️  Warning: OPENAI_API_KEY not set, will skip GPT models")

    if needs_anthropic and anthropic_key:
        has_valid_model = True
        print(f"✅ Anthropic API key loaded")
    elif needs_anthropic and not anthropic_key:
        print(f"⚠️  Warning: ANTHROPIC_API_KEY not set, will skip Claude models")

    if needs_gemini and gemini_key:
        has_valid_model = True
        print(f"✅ Gemini API key loaded")
    elif needs_gemini and not gemini_key:
        print(f"⚠️  Warning: GEMINI_API_KEY not set, will skip Gemini models")

    if not has_valid_model:
        print("\n❌ Error: No valid API keys for requested models")
        print("   Set API keys in .env file or export them:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export ANTHROPIC_API_KEY='your-key'")
        return
    print(f"📊 Testing on {args.limit} examples")
    print(f"🤖 Models: {', '.join(args.models)}")

    # Estimate cost
    cost_per_example = 0.01  # rough estimate
    estimated_cost = len(args.models) * args.limit * cost_per_example
    print(f"💰 Estimated cost: ${estimated_cost:.2f}")

    # Confirm
    if args.limit > 50:
        response = input("\n⚠️  This will test on >50 examples. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Create comparison
    comparison = ModelComparison()

    # Add requested models (lazy initialization)
    for model_key in args.models:
        if model_key == 'gpt4':
            if openai_key:
                comparison.add_model('GPT-4', OpenAIProvider(api_key=openai_key, model="gpt-4"))
            else:
                print(f"⚠️  Skipping GPT-4: OPENAI_API_KEY not set")
        elif model_key == 'gpt4o':
            if openai_key:
                comparison.add_model('GPT-4o', OpenAIProvider(api_key=openai_key, model="gpt-4o"))
            else:
                print(f"⚠️  Skipping GPT-4o: OPENAI_API_KEY not set")
        elif model_key == 'gpt4-turbo':
            if openai_key:
                comparison.add_model('GPT-4-Turbo', OpenAIProvider(api_key=openai_key, model="gpt-4-turbo"))
            else:
                print(f"⚠️  Skipping GPT-4-Turbo: OPENAI_API_KEY not set")
        elif model_key == 'gpt5':
            if openai_key:
                comparison.add_model('GPT-5', OpenAIProvider(api_key=openai_key, model="gpt-5"))
            else:
                print(f"⚠️  Skipping GPT-5: OPENAI_API_KEY not set")
        elif model_key == 'claude':
            if anthropic_key:
                comparison.add_model('Claude Sonnet', AnthropicProvider(api_key=anthropic_key, model="claude-3-5-sonnet-20241022"))
            else:
                print(f"⚠️  Skipping Claude Sonnet: ANTHROPIC_API_KEY not set")
        elif model_key == 'claude-opus':
            if anthropic_key:
                comparison.add_model('Claude Opus', AnthropicProvider(api_key=anthropic_key, model="claude-3-opus-20240229"))
            else:
                print(f"⚠️  Skipping Claude Opus: ANTHROPIC_API_KEY not set")
        elif model_key == 'claude-haiku':
            if anthropic_key:
                comparison.add_model('Claude Haiku', AnthropicProvider(api_key=anthropic_key, model="claude-3-haiku-20240307"))
            else:
                print(f"⚠️  Skipping Claude Haiku: ANTHROPIC_API_KEY not set")
        elif model_key == 'gemini':
            if gemini_key:
                comparison.add_model('Gemini 2.0 Flash', GeminiProvider(api_key=gemini_key, model="gemini-2.0-flash-exp"))
            else:
                print(f"⚠️  Skipping Gemini: GEMINI_API_KEY not set")
        elif model_key == 'gemini-pro':
            if gemini_key:
                comparison.add_model('Gemini 2.5 Pro', GeminiProvider(api_key=gemini_key, model="gemini-2.5-pro-002"))
            else:
                print(f"⚠️  Skipping Gemini Pro: GEMINI_API_KEY not set")
        else:
            print(f"⚠️  Unknown model: {model_key}")

    # Check if any models were added
    if not comparison.models:
        print("\n❌ Error: No models were added to comparison")
        print("   Check that API keys are set for at least one model")
        return

    # Run comparison
    try:
        results = comparison.run_contradiction_benchmark(limit=args.limit, verbose=args.verbose)
        comparison.print_comparison_report(results)

        print("\n" + "=" * 80)
        print("✅ COMPARISON COMPLETE")
        print("=" * 80)
        print("\nResults saved to evaluation/reports/")
        print("\nTo run again with different settings:")
        print(f"  python run_comparison.py --limit {args.limit * 2}  # More examples")
        print(f"  python run_comparison.py --limit {args.limit} --verbose  # Show details")
        print(f"  python run_comparison.py --models gpt4 claude claude-opus  # Add more models")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
