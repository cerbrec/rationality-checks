#!/usr/bin/env python3
"""
HaluEval Sample Evaluation
==========================

Evaluate our verification skills on a sample of HaluEval dataset.

HaluEval Dataset:
- 35,000 ChatGPT-generated hallucinated samples
- Human-annotated ground truth labels
- Categories: QA, Dialogue, Summarization

This script:
1. Downloads/loads a sample of HaluEval
2. Runs verification skills on each claim
3. Measures detection accuracy (precision, recall, F1)
4. Analyzes which types of hallucinations we catch best
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import requests
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.unified_orchestrator.verification_skills import (
    FactCheckingSkill,
    WorldStateVerificationSkill,
    EmpiricalTestingSkill,
    SynthesisSkill
)


# ============================================================================
# HALUEVAL DATA LOADING
# ============================================================================

def download_halueval_sample(
    category: str = "qa",
    sample_size: int = 50,
    output_dir: str = "evaluation/datasets/hallucination"
) -> str:
    """
    Download a sample of HaluEval dataset.

    Args:
        category: "qa", "dialogue", or "summarization"
        sample_size: Number of samples to download
        output_dir: Where to save the dataset

    Returns:
        str: Path to downloaded file
    """
    # HaluEval GitHub raw data URLs
    base_url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data"

    category_files = {
        "qa": "qa_data.json",
        "dialogue": "dialogue_data.json",
        "summarization": "summarization_data.json"
    }

    if category not in category_files:
        raise ValueError(f"Category must be one of: {list(category_files.keys())}")

    filename = category_files[category]
    url = f"{base_url}/{filename}"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / filename

    print(f"Downloading HaluEval {category} data from GitHub...")
    print(f"  URL: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse full dataset - HaluEval uses JSONL (JSON Lines) format
        text = response.text
        full_data = []

        for line in text.strip().split('\n'):
            if line.strip():
                try:
                    full_data.append(json.loads(line))
                except json.JSONDecodeError:
                    # If JSONL parsing fails, try as single JSON
                    continue

        # If JSONL parsing didn't work, try as single JSON object/array
        if not full_data:
            try:
                full_data = response.json()
                if isinstance(full_data, dict):
                    full_data = full_data.get("data", [full_data])
                elif not isinstance(full_data, list):
                    full_data = [full_data]
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse as JSON or JSONL: {e}")

        # Sample from it
        import random
        sample_data = random.sample(full_data, min(sample_size, len(full_data)))

        # Save sample
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=2)

        print(f"  ✓ Downloaded {len(sample_data)} samples to {output_path}")
        return str(output_path)

    except requests.RequestException as e:
        print(f"  ✗ Download failed: {e}")
        print(f"\nAlternative: Manually download from:")
        print(f"  https://github.com/RUCAIBox/HaluEval")
        raise


def load_halueval_data(file_path: str) -> List[Dict[str, Any]]:
    """Load HaluEval data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Normalize format
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "data" in data:
        return data["data"]
    else:
        return [data]


def parse_halueval_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse a HaluEval item into standard format.

    HaluEval format has both hallucinated and correct answers.
    We return BOTH to test our detection on each.

    Returns:
        List of parsed items (one for hallucinated, one for correct answer)
    """
    results = []

    # QA format - has question, hallucinated_answer, right_answer
    if "question" in item and "hallucinated_answer" in item:
        # Test the hallucinated answer
        results.append({
            "claim": f"Q: {item['question']}\nA: {item['hallucinated_answer']}",
            "is_hallucinated": True,  # This is ALWAYS a hallucination
            "category": "qa",
            "context": item.get("knowledge", ""),
            "answer_type": "hallucinated",
            "raw": item
        })

        # Test the correct answer
        if "right_answer" in item:
            results.append({
                "claim": f"Q: {item['question']}\nA: {item['right_answer']}",
                "is_hallucinated": False,  # This is ALWAYS correct
                "category": "qa",
                "context": item.get("knowledge", ""),
                "answer_type": "correct",
                "raw": item
            })

    # Dialogue format
    elif "dialogue" in item and "response" in item:
        # HaluEval dialogue has hallucinated response
        results.append({
            "claim": f"Context: {item['dialogue']}\nResponse: {item['response']}",
            "is_hallucinated": True,
            "category": "dialogue",
            "context": item.get("dialogue", ""),
            "answer_type": "hallucinated",
            "raw": item
        })

    # Summarization format
    elif "document" in item and "summary" in item:
        # HaluEval summary is hallucinated
        results.append({
            "claim": f"Summary: {item['summary']}",
            "is_hallucinated": True,
            "category": "summarization",
            "context": item.get("document", ""),
            "answer_type": "hallucinated",
            "raw": item
        })

    # Generic format (fallback)
    else:
        results.append({
            "claim": str(item.get("text", item.get("claim", str(item)))),
            "is_hallucinated": item.get("is_hallucinated", item.get("hallucination", "no") == "yes"),
            "category": "generic",
            "context": "",
            "answer_type": "unknown",
            "raw": item
        })

    return results


# ============================================================================
# VERIFICATION
# ============================================================================

class SimpleLLMProvider:
    """Simple LLM provider wrapper for fact-checking - prefers Bedrock, falls back to Anthropic API"""

    def __init__(self):
        self.provider = None
        self.provider_type = None

        # Try Bedrock first (preferred)
        if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from src.verification_pipeline import BedrockProvider

                self.provider = BedrockProvider.from_env(
                    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
                )
                self.provider_type = "bedrock"
                print("  ✓ Using AWS Bedrock for LLM verification")
            except Exception as e:
                print(f"  ⚠️  Bedrock initialization failed: {e}")

        # Fall back to Anthropic API
        if not self.provider and os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                self.model = "claude-sonnet-4-20250514"
                self.provider_type = "anthropic"
                print("  ✓ Using Anthropic API for LLM verification")
            except Exception as e:
                print(f"  ⚠️  Anthropic API initialization failed: {e}")

        if not self.provider and not hasattr(self, 'client'):
            raise ValueError(
                "No LLM provider available. Set either:\n"
                "  - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (for Bedrock), or\n"
                "  - ANTHROPIC_API_KEY (for direct Anthropic API)"
            )

    def call(self, prompt: str, max_tokens: int = 500) -> str:
        """Call LLM and return response text"""
        if self.provider_type == "bedrock":
            # Use BedrockProvider.generate()
            return self.provider.generate(prompt)

        elif self.provider_type == "anthropic":
            # Use Anthropic API directly
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract text from response
            text_content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    text_content += block.text

            return text_content

        else:
            raise ValueError("No LLM provider initialized")


class HaluEvalVerifier:
    """Verifies HaluEval claims using our verification skills"""

    def __init__(self, enable_web_search: bool = False):
        """
        Initialize verifier.

        Args:
            enable_web_search: Enable web search (requires SERPER_API_KEY)
        """
        # Initialize LLM provider for fact-checking (Bedrock preferred, Anthropic API fallback)
        llm_provider = None
        try:
            llm_provider = SimpleLLMProvider()
        except ValueError as e:
            print(f"  ⚠️  {e}")
            print("      Fact-checking will use placeholder confidence values without LLM")

        self.fact_checker = FactCheckingSkill(llm_provider=llm_provider)
        self.world_state = WorldStateVerificationSkill()
        self.empirical_test = EmpiricalTestingSkill(llm_provider=llm_provider)
        self.synthesis = SynthesisSkill()
        self.enable_web_search = enable_web_search

        if enable_web_search and not os.getenv("SERPER_API_KEY"):
            print("  ⚠️  Web search disabled (SERPER_API_KEY not set)")
            self.enable_web_search = False

    def verify_claim(self, claim: str, context: str = "") -> Dict[str, Any]:
        """
        Verify a single claim.

        Args:
            claim: Claim to verify
            context: Optional context

        Returns:
            dict: Verification result with prediction
        """
        verification_results = []

        # 1. Fact checking with LLM plausibility check
        fact_result = self.fact_checker.execute(claim=claim, context=context)
        verification_results.append(fact_result)

        # 2. Empirical testing for logical consistency
        empirical_result = self.empirical_test.execute(claims=[claim], context=context)
        verification_results.append(empirical_result)

        # 3. Synthesize results
        synthesis_result = self.synthesis.execute(verification_results=verification_results)

        # Determine if hallucinated based on confidence threshold
        overall_confidence = synthesis_result.get("overall_confidence", 0.5)

        # Lower confidence = more likely hallucinated
        # Threshold: <0.6 = likely hallucinated
        predicted_hallucinated = overall_confidence < 0.6

        return {
            "claim": claim,
            "predicted_hallucinated": predicted_hallucinated,
            "confidence": overall_confidence,
            "verification_results": verification_results,
            "synthesis": synthesis_result
        }


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate precision, recall, F1 for hallucination detection"""

    true_positives = sum(
        1 for r in results
        if r["ground_truth"] and r["predicted_hallucinated"]
    )

    false_positives = sum(
        1 for r in results
        if not r["ground_truth"] and r["predicted_hallucinated"]
    )

    false_negatives = sum(
        1 for r in results
        if r["ground_truth"] and not r["predicted_hallucinated"]
    )

    true_negatives = sum(
        1 for r in results
        if not r["ground_truth"] and not r["predicted_hallucinated"]
    )

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(results) if results else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "total": len(results)
    }


def analyze_results(results: List[Dict[str, Any]]) -> None:
    """Analyze and print detailed results"""

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    metrics = calculate_metrics(results)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}  (of claims we flagged, how many were actually hallucinations)")
    print(f"  Recall:    {metrics['recall']:.1%}  (of actual hallucinations, how many did we catch)")
    print(f"  F1 Score:  {metrics['f1']:.1%}  (harmonic mean of precision and recall)")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']:3d}  (correctly detected hallucinations)")
    print(f"  False Positives: {metrics['false_positives']:3d}  (false alarms - flagged truth as hallucination)")
    print(f"  False Negatives: {metrics['false_negatives']:3d}  (missed hallucinations)")
    print(f"  True Negatives:  {metrics['true_negatives']:3d}  (correctly accepted truth)")
    print(f"  Total Samples:   {metrics['total']:3d}")

    # Analyze by category if available
    categories = defaultdict(list)
    for r in results:
        categories[r.get("category", "unknown")].append(r)

    if len(categories) > 1:
        print(f"\nResults by Category:")
        for category, cat_results in categories.items():
            cat_metrics = calculate_metrics(cat_results)
            print(f"  {category.capitalize():15s}: {cat_metrics['f1']:.1%} F1  ({len(cat_results)} samples)")

    # Analyze by answer type (hallucinated vs correct)
    answer_types = defaultdict(list)
    for r in results:
        answer_types[r.get("answer_type", "unknown")].append(r)

    if len(answer_types) > 1:
        print(f"\nResults by Answer Type:")
        for ans_type, ans_results in answer_types.items():
            ans_metrics = calculate_metrics(ans_results)
            print(f"  {ans_type.capitalize():15s}: {ans_metrics['accuracy']:.1%} accuracy  ({len(ans_results)} samples)")
            if ans_type == "hallucinated":
                print(f"                       Recall: {ans_metrics['recall']:.1%} (caught {ans_metrics['true_positives']}/{ans_metrics['true_positives'] + ans_metrics['false_negatives']} hallucinations)")
            elif ans_type == "correct":
                print(f"                       Specificity: {ans_metrics['true_negatives']}/{len(ans_results)} correctly accepted truth")

    # Show example errors
    false_positives = [r for r in results if not r["ground_truth"] and r["predicted_hallucinated"]]
    false_negatives = [r for r in results if r["ground_truth"] and not r["predicted_hallucinated"]]

    if false_positives:
        print(f"\nExample False Positive (flagged truth as hallucination):")
        fp = false_positives[0]
        print(f"  Claim: {fp['claim'][:150]}...")
        print(f"  Confidence: {fp['confidence']:.2f}")

    if false_negatives:
        print(f"\nExample False Negative (missed hallucination):")
        fn = false_negatives[0]
        print(f"  Claim: {fn['claim'][:150]}...")
        print(f"  Confidence: {fn['confidence']:.2f}")


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    """Run HaluEval evaluation"""

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate verification skills on HaluEval dataset sample"
    )
    parser.add_argument("--category", default="qa", choices=["qa", "dialogue", "summarization"],
                       help="HaluEval category to test")
    parser.add_argument("--sample-size", type=int, default=50,
                       help="Number of samples to test (default: 50)")
    parser.add_argument("--enable-web-search", action="store_true",
                       help="Enable web search for fact-checking (requires SERPER_API_KEY)")
    parser.add_argument("--data-file", help="Use existing data file instead of downloading")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed verification progress")

    args = parser.parse_args()

    print("=" * 80)
    print("HALUEVAL EVALUATION - Verification Skills Test")
    print("=" * 80)
    print(f"\nCategory: {args.category}")
    print(f"Sample Size: {args.sample_size}")
    print(f"Web Search: {'Enabled' if args.enable_web_search else 'Disabled'}")

    # Load or download data
    if args.data_file:
        print(f"\nLoading data from: {args.data_file}")
        data_file = args.data_file
    else:
        print(f"\nDownloading HaluEval sample...")
        try:
            data_file = download_halueval_sample(
                category=args.category,
                sample_size=args.sample_size
            )
        except Exception as e:
            print(f"\n❌ Failed to download: {e}")
            print("\nTo use a local file instead:")
            print(f"  python evaluate_halueval.py --data-file path/to/data.json")
            return 1

    # Load data
    raw_data = load_halueval_data(data_file)
    print(f"\nLoaded {len(raw_data)} samples")

    # Parse data - each item can produce multiple test cases (hallucinated + correct)
    parsed_data = []
    for item in raw_data:
        parsed_data.extend(parse_halueval_item(item))

    print(f"Generated {len(parsed_data)} test cases ({len(raw_data)} original samples x ~2 answers each)")

    # Initialize verifier
    print(f"\nInitializing verifier...")
    verifier = HaluEvalVerifier(enable_web_search=args.enable_web_search)

    # Run verification
    print(f"\nVerifying {len(parsed_data)} claims...")
    if args.verbose:
        print(f"  (Using LLM plausibility checking)")
    results = []

    for i, item in enumerate(parsed_data, 1):
        if not args.verbose:
            print(f"  [{i}/{len(parsed_data)}] Verifying...", end='\r')

        verification = verifier.verify_claim(item["claim"], item["context"])

        result = {
            "claim": item["claim"][:200],  # Truncate for storage
            "category": item["category"],
            "answer_type": item.get("answer_type", "unknown"),
            "ground_truth": item["is_hallucinated"],
            "predicted_hallucinated": verification["predicted_hallucinated"],
            "confidence": verification["confidence"],
            "correct": item["is_hallucinated"] == verification["predicted_hallucinated"]
        }

        if args.verbose and i == 1:
            # Print first verification details for debugging
            print(f"\nFirst verification details:")
            print(f"  Claim: {item['claim'][:100]}...")
            print(f"  Answer type: {item.get('answer_type')}")
            print(f"  Confidence: {verification['confidence']:.2f}")
            print(f"  Verification results: {len(verification.get('verification_results', []))} checks")
            for vr in verification.get('verification_results', []):
                print(f"    - {vr.get('method', 'unknown')}: {vr.get('confidence', 0):.2f}")

        results.append(result)

    print()  # New line after progress

    # Analyze results
    analyze_results(results)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "category": args.category,
                    "sample_size": len(results),
                    "web_search_enabled": args.enable_web_search
                },
                "metrics": calculate_metrics(results),
                "results": results
            }, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
