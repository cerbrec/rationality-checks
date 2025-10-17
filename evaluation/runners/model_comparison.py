"""
Model Comparison Framework

Compares different LLM providers (OpenAI vs Claude) on the same benchmarks.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integrated_verification import IntegratedVerificationPipeline
from verification_pipeline import LLMProvider
from evaluation.benchmarks.contradiction_benchmark import ContradictionBenchmark
from evaluation.metrics.accuracy_metrics import MetricResult


@dataclass
class ModelPerformance:
    """Performance metrics for a single model"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    calibration_error: float
    total_time_seconds: float
    avg_time_per_example: float
    total_examples: int
    by_label_accuracy: Dict[str, float]


class ModelComparison:
    """
    Compare multiple LLM providers on the same benchmarks.

    Usage:
        comparison = ModelComparison()
        comparison.add_model("GPT-4", openai_provider)
        comparison.add_model("Claude Sonnet", claude_provider)
        results = comparison.run_contradiction_benchmark(limit=50)
        comparison.print_comparison_report(results)
    """

    def __init__(self):
        self.models: Dict[str, LLMProvider] = {}

    def add_model(self, name: str, provider: LLMProvider):
        """Add a model to compare"""
        self.models[name] = provider
        print(f"✅ Added model: {name}")

    def run_contradiction_benchmark(
        self,
        limit: int = 50,
        verbose: bool = False,
        datasets: List[str] = None
    ) -> Dict[str, Dict]:
        """
        Run contradiction detection benchmark on all models.

        Args:
            limit: Number of examples to test per dataset
            verbose: Print detailed progress
            datasets: List of datasets to test on. If None, uses all datasets.
                     Options: 'anli', 'snli', 'scitail', 'vitaminc'

        Returns:
            Dictionary mapping model name to results
        """
        if not self.models:
            raise ValueError("No models added. Use add_model() first.")

        # Default to all datasets
        if datasets is None:
            datasets = ['anli', 'snli', 'scitail', 'vitaminc']
        elif isinstance(datasets, str):
            datasets = [datasets]

        print("\n" + "=" * 80)
        print("RUNNING MODEL COMPARISON")
        print("=" * 80)
        print(f"Models: {', '.join(self.models.keys())}")
        print(f"Datasets: {', '.join(d.upper() for d in datasets)}")
        print(f"Examples per dataset: {limit}")
        print("=" * 80)

        all_results = {}

        for model_name, provider in self.models.items():
            print(f"\n\n{'=' * 80}")
            print(f"TESTING: {model_name}")
            print("=" * 80)

            # Store results for all datasets
            model_dataset_results = {}

            for dataset_name in datasets:
                print(f"\n📊 Dataset: {dataset_name.upper()}")
                print("-" * 80)

                # Create pipeline for this model
                pipeline = IntegratedVerificationPipeline(provider)
                benchmark = ContradictionBenchmark(pipeline, dataset_name=dataset_name)

                # Run benchmark and time it
                start_time = time.time()
                try:
                    results = benchmark.run_benchmark(limit=limit, verbose=verbose)
                    elapsed_time = time.time() - start_time

                    # Extract metrics
                    metrics = results['metrics']
                    by_label = results['by_label_accuracy']

                    # Convert by_label to simple accuracy dict
                    by_label_acc = {
                        label: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                        for label, stats in by_label.items()
                    }

                    # Create performance summary
                    performance = ModelPerformance(
                        model_name=f"{model_name} ({dataset_name.upper()})",
                        accuracy=metrics['accuracy'].value,
                        precision=metrics['precision'].value,
                        recall=metrics['recall'].value,
                        f1_score=metrics['f1'].value,
                        calibration_error=metrics.get('calibration', MetricResult('', 0.0)).value,
                        total_time_seconds=elapsed_time,
                        avg_time_per_example=elapsed_time / limit,
                        total_examples=limit,
                        by_label_accuracy=by_label_acc
                    )

                    model_dataset_results[dataset_name] = {
                        'performance': performance,
                        'detailed_results': results['results'],
                        'metrics': metrics
                    }

                    print(f"⏱️  Time: {elapsed_time:.1f}s ({elapsed_time/limit:.2f}s per example)")

                except Exception as e:
                    print(f"❌ Error on {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            all_results[model_name] = model_dataset_results

        return all_results

    def print_comparison_report(self, results: Dict[str, Dict]):
        """Print a detailed comparison report"""
        print("\n\n" + "=" * 80)
        print("MODEL COMPARISON REPORT")
        print("=" * 80)

        # Flatten results: model -> dataset -> performance into list of performances
        performances = []
        for model_name, dataset_results in results.items():
            for dataset_name, result_data in dataset_results.items():
                performances.append(result_data['performance'])

        if not performances:
            print("No results to compare")
            return

        # Overall metrics comparison
        print("\n📊 OVERALL METRICS")
        print("-" * 80)
        # Truncate model names to fit
        model_names = [p.model_name[:25] for p in performances]
        print(f"{'Metric':<20} " + " ".join(f"{name:>27}" for name in model_names))
        print("-" * 80)

        metrics_to_compare = [
            ('Accuracy', 'accuracy'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('F1 Score', 'f1_score'),
            ('Calibration Err', 'calibration_error'),
        ]

        for metric_name, attr in metrics_to_compare:
            values = [getattr(p, attr) for p in performances]
            best_idx = values.index(max(values)) if attr != 'calibration_error' else values.index(min(values))

            print(f"{metric_name:<20} ", end="")
            for i, value in enumerate(values):
                marker = " 🏆" if i == best_idx else "   "
                print(f"{value:>27.4f}{marker}", end=" ")
            print()

        # Efficiency metrics
        print("\n⚡ EFFICIENCY METRICS")
        print("-" * 80)
        print(f"{'Metric':<20} " + " ".join(f"{name:>27}" for name in model_names))
        print("-" * 80)

        efficiency_metrics = [
            ('Total Time (s)', 'total_time_seconds'),
            ('Time per Example (s)', 'avg_time_per_example'),
        ]

        for metric_name, attr in efficiency_metrics:
            values = [getattr(p, attr) for p in performances]
            best_idx = values.index(min(values))

            print(f"{metric_name:<20} ", end="")
            for i, value in enumerate(values):
                marker = " 🏆" if i == best_idx else "   "
                print(f"{value:>27.2f}{marker}", end=" ")
            print()

        # Performance by label type
        print("\n🏷️  ACCURACY BY LABEL TYPE")
        print("-" * 80)

        # Get all unique labels
        all_labels = set()
        for p in performances:
            all_labels.update(p.by_label_accuracy.keys())

        print(f"{'Label':<20} " + " ".join(f"{name:>27}" for name in model_names))
        print("-" * 80)

        for label in sorted(all_labels):
            print(f"{label:<20} ", end="")
            values = [p.by_label_accuracy.get(label, 0.0) for p in performances]
            best_idx = values.index(max(values)) if max(values) > 0 else -1

            for i, value in enumerate(values):
                marker = " 🏆" if i == best_idx and best_idx >= 0 else "   "
                print(f"{value:>27.4f}{marker}", end=" ")
            print()

        # Summary recommendation
        print("\n💡 SUMMARY")
        print("-" * 80)

        best_overall = max(performances, key=lambda p: p.f1_score)
        fastest = min(performances, key=lambda p: p.avg_time_per_example)
        best_calibration = min(performances, key=lambda p: p.calibration_error)

        print(f"Best Overall (F1): {best_overall.model_name} ({best_overall.f1_score:.4f})")
        print(f"Fastest: {fastest.model_name} ({fastest.avg_time_per_example:.2f}s/example)")
        print(f"Best Calibrated: {best_calibration.model_name} (ECE: {best_calibration.calibration_error:.4f})")

        # Save results to JSON
        output_dir = Path("evaluation/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"comparison_{timestamp}.json"

        comparison_data = {
            'timestamp': timestamp,
            'models': [asdict(p) for p in performances],
            'benchmark': 'contradiction_detection_multi_dataset',
            'num_examples_per_dataset': performances[0].total_examples
        }

        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")


def main():
    """Example usage: Compare OpenAI and Claude"""
    import os
    from verification_pipeline import OpenAIProvider, AnthropicProvider

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key or not anthropic_key:
        print("⚠️  Error: Both OPENAI_API_KEY and ANTHROPIC_API_KEY must be set")
        print("\nSet them in your .env file or environment:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return

    # Create comparison
    comparison = ModelComparison()

    # Add models
    print("\n🔧 Setting up models...")
    comparison.add_model("GPT-4", OpenAIProvider(api_key=openai_key, model="gpt-4"))
    comparison.add_model("Claude Sonnet", AnthropicProvider(
        api_key=anthropic_key,
        model="claude-3-5-sonnet-20241022"
    ))

    # Run comparison (start with small sample)
    results = comparison.run_contradiction_benchmark(limit=20, verbose=False)

    # Print report
    comparison.print_comparison_report(results)


if __name__ == "__main__":
    main()
