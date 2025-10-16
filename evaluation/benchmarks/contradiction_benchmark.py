"""
Contradiction Detection Benchmark using ANLI dataset

Tests whether the rationality-llm pipeline can correctly identify
contradictions between premise and hypothesis statements.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add parent directory to path to import from rationality-llm
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integrated_verification import IntegratedVerificationPipeline
from evaluation.metrics.accuracy_metrics import (
    evaluate_contradiction_detection,
    print_evaluation_report
)


@dataclass
class ContradictionExample:
    """Single example for contradiction detection"""
    id: str
    premise: str
    hypothesis: str
    label: int  # 0=entailment, 1=neutral, 2=contradiction
    label_name: str


class ContradictionBenchmark:
    """
    Benchmark for testing contradiction detection on multiple datasets.

    The pipeline is given a document containing both premise and hypothesis.
    We check if it correctly identifies contradictions.

    Supported datasets:
    - ANLI: Adversarial Natural Language Inference (300 samples)
    - SNLI: Stanford Natural Language Inference (100 samples)
    - SciTail: Scientific entailment reasoning (100 samples)
    - VitaminC: Fact verification dataset (100 samples)
    """

    def __init__(self, pipeline: IntegratedVerificationPipeline, dataset_name: str = "anli"):
        self.pipeline = pipeline
        self.dataset_name = dataset_name.lower()

        # Map dataset names to file paths
        dataset_files = {
            'anli': 'evaluation/datasets/anli_samples.json',
            'snli': 'evaluation/datasets/snli_samples.json',
            'scitail': 'evaluation/datasets/scitail_samples.json',
            'vitaminc': 'evaluation/datasets/vitaminc_samples.json'
        }

        if self.dataset_name not in dataset_files:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(dataset_files.keys())}")

        self.dataset_path = Path(dataset_files[self.dataset_name])

    def _normalize_label(self, label, label_name=None) -> Tuple[int, str]:
        """
        Normalize labels from different datasets to standard format.
        Returns: (numeric_label, label_name)
        - 0 = entailment
        - 1 = neutral / not enough info
        - 2 = contradiction / refutes
        """
        # If already numeric, just normalize label_name
        if isinstance(label, int):
            names = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
            return label, label_name or names.get(label, 'unknown')

        # String label normalization
        label_str = str(label).lower()

        # SciTail format
        if label_str == 'entailment':
            return 0, 'entailment'
        elif label_str == 'neutral':
            return 1, 'neutral'

        # VitaminC format
        elif label_str == 'supports':
            return 0, 'entailment'
        elif label_str == 'not enough info':
            return 1, 'neutral'
        elif label_str == 'refutes':
            return 2, 'contradiction'

        else:
            raise ValueError(f"Unknown label format: {label}")

    def load_dataset(self, limit: int = None) -> List[ContradictionExample]:
        """Load dataset with support for different formats"""
        with open(self.dataset_path) as f:
            data = json.load(f)

        if limit:
            data = data[:limit]

        examples = []
        for item in data:
            # VitaminC uses claim/evidence instead of premise/hypothesis
            if self.dataset_name == 'vitaminc':
                premise = item['evidence']
                hypothesis = item['claim']
            else:
                premise = item['premise']
                hypothesis = item['hypothesis']

            # Normalize labels
            label_raw = item['label']
            label_name_raw = item.get('label_name')
            label, label_name = self._normalize_label(label_raw, label_name_raw)

            examples.append(ContradictionExample(
                id=item['id'],
                premise=premise,
                hypothesis=hypothesis,
                label=label,
                label_name=label_name
            ))

        return examples

    def run_single_example(self, example: ContradictionExample) -> Dict:
        """
        Run pipeline on a single example.

        We create a document that states both the premise and hypothesis,
        then check if the pipeline detects an inconsistency.
        """
        # Create a document with both claims
        document = f"""
Analysis:

Premise: {example.premise}

Hypothesis: {example.hypothesis}

Based on the premise, the hypothesis {'contradicts' if example.label == 2 else 'is consistent with' if example.label == 0 else 'may or may not be true given'} the stated facts.
"""

        # Run verification
        report = self.pipeline.verify_analysis(
            original_output=document,
            original_query="Verify logical consistency between premise and hypothesis"
        )

        # Check if pipeline detected a contradiction
        has_contradiction = False
        max_confidence = 0.0

        for assessment in report.assessments:
            # Look for failed verifications with low confidence
            if assessment.recommendation in ['revise', 'remove']:
                has_contradiction = True
                max_confidence = max(max_confidence, 1.0 - assessment.overall_confidence)
            elif any(not result.passed for result in assessment.verification_results):
                has_contradiction = True
                max_confidence = max(max_confidence, 0.8)

        # Ground truth: contradiction exists if label == 2
        ground_truth_contradiction = (example.label == 2)

        return {
            'example_id': example.id,
            'predicted_contradiction': has_contradiction,
            'ground_truth_contradiction': ground_truth_contradiction,
            'confidence': max_confidence if has_contradiction else 0.5,
            'label_name': example.label_name,
            'num_claims_extracted': len(report.extracted_claims),
            'report': report
        }

    def run_benchmark(self, limit: int = 50, verbose: bool = False) -> Dict:
        """
        Run benchmark on dataset.

        Args:
            limit: Maximum number of examples to test
            verbose: Print detailed results for each example

        Returns:
            Dictionary with results and metrics
        """
        print("=" * 80)
        print(f"CONTRADICTION DETECTION BENCHMARK ({self.dataset_name.upper()})")
        print("=" * 80)

        examples = self.load_dataset(limit=limit)
        print(f"\nTesting on {len(examples)} examples from {self.dataset_name.upper()} dataset...")

        results = []
        predictions = []
        ground_truths = []
        confidences = []

        for i, example in enumerate(examples):
            if verbose or (i + 1) % 10 == 0:
                print(f"\nProcessing {i + 1}/{len(examples)}: {example.id}")

            result = self.run_single_example(example)
            results.append(result)

            predictions.append(result['predicted_contradiction'])
            ground_truths.append(result['ground_truth_contradiction'])
            confidences.append(result['confidence'])

            if verbose:
                correct = "✓" if result['predicted_contradiction'] == result['ground_truth_contradiction'] else "✗"
                print(f"  {correct} Predicted: {result['predicted_contradiction']}, "
                      f"Actual: {result['ground_truth_contradiction']} ({result['label_name']})")

        # Calculate metrics
        metrics = evaluate_contradiction_detection(
            predictions,
            ground_truths,
            confidences
        )

        # Print report
        print_evaluation_report(metrics)

        # Additional statistics
        print("\n" + "=" * 80)
        print("ADDITIONAL STATISTICS")
        print("=" * 80)

        # Performance by label type
        by_label = {}
        for result in results:
            label = result['label_name']
            if label not in by_label:
                by_label[label] = {'correct': 0, 'total': 0}
            by_label[label]['total'] += 1
            if result['predicted_contradiction'] == result['ground_truth_contradiction']:
                by_label[label]['correct'] += 1

        print("\nAccuracy by label type:")
        for label, stats in by_label.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {label}: {acc:.3f} ({stats['correct']}/{stats['total']})")

        avg_claims = sum(r['num_claims_extracted'] for r in results) / len(results)
        print(f"\nAverage claims extracted per example: {avg_claims:.1f}")

        return {
            'metrics': metrics,
            'results': results,
            'by_label_accuracy': by_label
        }


def main():
    """Example usage"""
    from verification_pipeline import AnthropicProvider
    import os

    # Initialize pipeline
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    llm = AnthropicProvider(api_key=api_key)
    pipeline = IntegratedVerificationPipeline(llm)

    # Run benchmark
    benchmark = ContradictionBenchmark(pipeline)
    results = benchmark.run_benchmark(limit=10, verbose=True)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
