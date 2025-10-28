#!/usr/bin/env python3
"""
HaluEval Single-Prompt Baseline
================================

Compare our multi-step verification pipeline to a simple single-prompt baseline
using the SAME model (Claude Sonnet 4.5 via Bedrock).

This tests whether the multi-step pipeline (FactChecking + Empirical + Synthesis)
actually adds value compared to just asking Claude: "Is this hallucinated?"
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.verification_pipeline import BedrockProvider


class SinglePromptBaseline:
    """Simple baseline: One prompt asking if claim is hallucinated"""

    def __init__(self):
        self.llm = BedrockProvider.from_env(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        print("  ✓ Using AWS Bedrock (Sonnet 4.5) for baseline")

    def verify_claim(self, claim: str, context: str = "") -> Dict[str, Any]:
        """
        Single-prompt hallucination detection.

        Args:
            claim: Claim to verify
            context: Optional context

        Returns:
            dict: Verification result with prediction
        """
        context_section = f"\n\nCONTEXT:\n{context}" if context else ""

        prompt = f"""You are evaluating whether a claim contains hallucinated (false/incorrect) information.

CLAIM:
{claim}
{context_section}

Is this claim hallucinated (contains false information)?

Respond in this exact format:
HALLUCINATED: <yes/no>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>"""

        response = self.llm.generate(prompt)

        # Extract answer
        import re
        hallucinated_match = re.search(r'HALLUCINATED:\s*(yes|no)', response, re.IGNORECASE)
        confidence_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0)', response)

        predicted_hallucinated = hallucinated_match.group(1).lower() == 'yes' if hallucinated_match else False
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5

        return {
            "claim": claim,
            "predicted_hallucinated": predicted_hallucinated,
            "confidence": confidence,
            "raw_response": response
        }


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate precision, recall, F1"""
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


def main():
    """Run single-prompt baseline on same HaluEval data"""

    # Load the SAME test data we used for multi-step evaluation
    test_data_file = "evaluation/datasets/hallucination/qa_data.json"

    if not os.path.exists(test_data_file):
        print("❌ Error: Test data not found. Run evaluate_halueval.py first.")
        return 1

    # Load test data
    with open(test_data_file, 'r') as f:
        raw_data = json.load(f)

    print("=" * 80)
    print("SINGLE-PROMPT BASELINE TEST")
    print("=" * 80)
    print(f"\nModel: Claude Sonnet 4.5 (via Bedrock)")
    print(f"Method: Single prompt asking 'Is this hallucinated?'")
    print(f"Test Set: Same {len(raw_data)} samples as multi-step evaluation")

    # Parse data (from evaluate_halueval.py logic)
    from evaluate_halueval import parse_halueval_item

    parsed_data = []
    for item in raw_data:
        parsed_data.extend(parse_halueval_item(item))

    print(f"\nGenerated {len(parsed_data)} test cases")

    # Initialize baseline
    print(f"\nInitializing single-prompt baseline...")
    baseline = SinglePromptBaseline()

    # Run verification
    print(f"\nVerifying {len(parsed_data)} claims...")
    results = []

    for i, item in enumerate(parsed_data, 1):
        print(f"  [{i}/{len(parsed_data)}] Verifying...", end='\r')

        verification = baseline.verify_claim(item["claim"], item["context"])

        result = {
            "claim": item["claim"][:200],
            "category": item["category"],
            "answer_type": item.get("answer_type", "unknown"),
            "ground_truth": item["is_hallucinated"],
            "predicted_hallucinated": verification["predicted_hallucinated"],
            "confidence": verification["confidence"],
            "correct": item["is_hallucinated"] == verification["predicted_hallucinated"]
        }

        results.append(result)

    print()  # New line after progress

    # Calculate metrics
    metrics = calculate_metrics(results)

    print("\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1 Score:  {metrics['f1']:.1%}")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']:3d}")
    print(f"  False Positives: {metrics['false_positives']:3d}")
    print(f"  False Negatives: {metrics['false_negatives']:3d}")
    print(f"  True Negatives:  {metrics['true_negatives']:3d}")
    print(f"  Total Samples:   {metrics['total']:3d}")

    # Save results
    output_path = "halueval_baseline_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            "metadata": {
                "method": "single_prompt_baseline",
                "model": "claude-sonnet-4-5",
                "sample_size": len(results)
            },
            "metrics": metrics,
            "results": results
        }, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Baseline complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
