"""
Accuracy metrics for evaluating contradiction detection and claim verification
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class MetricResult:
    """Results from a metric calculation"""
    metric_name: str
    value: float
    details: Dict = None


def calculate_accuracy(predictions: List[bool], ground_truth: List[bool]) -> float:
    """Calculate simple accuracy"""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions)


def calculate_precision_recall_f1(
    predictions: List[bool],
    ground_truth: List[bool]
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score for binary classification.

    Args:
        predictions: List of predicted labels (True = positive, False = negative)
        ground_truth: List of ground truth labels

    Returns:
        Tuple of (precision, recall, f1)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    if len(predictions) == 0:
        return 0.0, 0.0, 0.0

    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def calculate_confusion_matrix(
    predictions: List[int],
    ground_truth: List[int],
    num_classes: int = 3
) -> np.ndarray:
    """
    Calculate confusion matrix for multi-class classification.

    Args:
        predictions: List of predicted class labels (0, 1, 2, ...)
        ground_truth: List of ground truth labels
        num_classes: Number of classes

    Returns:
        Confusion matrix as numpy array
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for pred, true in zip(predictions, ground_truth):
        if 0 <= pred < num_classes and 0 <= true < num_classes:
            matrix[true, pred] += 1

    return matrix


def calculate_calibration_error(
    confidences: List[float],
    correctness: List[bool],
    num_bins: int = 10
) -> Dict:
    """
    Calculate Expected Calibration Error (ECE).

    A well-calibrated model should have predictions where:
    - Claims with 90% confidence are correct 90% of the time
    - Claims with 50% confidence are correct 50% of the time, etc.

    Args:
        confidences: List of confidence scores (0.0 to 1.0)
        correctness: List of whether predictions were correct (True/False)
        num_bins: Number of bins to divide confidence scores into

    Returns:
        Dictionary with calibration metrics
    """
    if len(confidences) != len(correctness):
        raise ValueError("Confidences and correctness must have same length")

    if len(confidences) == 0:
        return {'ece': 0.0, 'bins': []}

    bins = []
    bin_edges = np.linspace(0, 1, num_bins + 1)

    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower, bin_upper = bin_edges[i], bin_edges[i + 1]

        # Find samples in this bin
        in_bin = [
            (conf, corr)
            for conf, corr in zip(confidences, correctness)
            if bin_lower <= conf < bin_upper or (i == num_bins - 1 and conf == 1.0)
        ]

        if not in_bin:
            continue

        # Calculate accuracy and average confidence for this bin
        bin_confidences = [conf for conf, _ in in_bin]
        bin_correctness = [corr for _, corr in in_bin]

        avg_confidence = np.mean(bin_confidences)
        avg_accuracy = np.mean(bin_correctness)
        bin_weight = len(in_bin) / total_samples

        # Calibration error for this bin
        bin_error = abs(avg_confidence - avg_accuracy)
        ece += bin_weight * bin_error

        bins.append({
            'range': f'{bin_lower:.2f}-{bin_upper:.2f}',
            'count': len(in_bin),
            'avg_confidence': avg_confidence,
            'avg_accuracy': avg_accuracy,
            'calibration_error': bin_error
        })

    return {
        'ece': ece,  # Expected Calibration Error
        'bins': bins
    }


def evaluate_contradiction_detection(
    predictions: List[bool],
    ground_truth: List[bool],
    confidences: List[float] = None
) -> Dict[str, MetricResult]:
    """
    Comprehensive evaluation for contradiction detection task.

    Args:
        predictions: List of whether model detected contradiction (True/False)
        ground_truth: List of whether contradiction actually exists
        confidences: Optional list of confidence scores

    Returns:
        Dictionary of metric results
    """
    results = {}

    # Accuracy
    accuracy = calculate_accuracy(predictions, ground_truth)
    results['accuracy'] = MetricResult(
        metric_name='accuracy',
        value=accuracy
    )

    # Precision, Recall, F1
    precision, recall, f1 = calculate_precision_recall_f1(predictions, ground_truth)
    results['precision'] = MetricResult(
        metric_name='precision',
        value=precision,
        details={'description': 'Of detected contradictions, how many were real?'}
    )
    results['recall'] = MetricResult(
        metric_name='recall',
        value=recall,
        details={'description': 'Of real contradictions, how many were detected?'}
    )
    results['f1'] = MetricResult(
        metric_name='f1_score',
        value=f1
    )

    # Calibration (if confidences provided)
    if confidences is not None:
        correctness = [p == g for p, g in zip(predictions, ground_truth)]
        calibration = calculate_calibration_error(confidences, correctness)
        results['calibration'] = MetricResult(
            metric_name='calibration_error',
            value=calibration['ece'],
            details={'bins': calibration['bins']}
        )

    return results


def print_evaluation_report(metrics: Dict[str, MetricResult]):
    """Print a formatted evaluation report"""
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    for metric_name, result in metrics.items():
        print(f"\n{result.metric_name.upper()}: {result.value:.4f}")
        if result.details:
            if 'description' in result.details:
                print(f"  {result.details['description']}")
            if 'bins' in result.details:
                print("  Calibration by confidence bins:")
                for bin_info in result.details['bins']:
                    print(f"    {bin_info['range']}: "
                          f"conf={bin_info['avg_confidence']:.3f}, "
                          f"acc={bin_info['avg_accuracy']:.3f}, "
                          f"n={bin_info['count']}")


if __name__ == "__main__":
    # Example usage
    print("Example: Contradiction Detection Evaluation")

    # Simulate some predictions
    predictions = [True, False, True, True, False, True, False, False]
    ground_truth = [True, False, False, True, False, True, False, True]
    confidences = [0.9, 0.8, 0.6, 0.95, 0.7, 0.85, 0.75, 0.5]

    metrics = evaluate_contradiction_detection(
        predictions,
        ground_truth,
        confidences
    )

    print_evaluation_report(metrics)
