# LLM Evaluation Framework

Comprehensive evaluation system for comparing different LLMs (OpenAI GPT-4 vs Claude) on logical inconsistency detection and claim verification tasks.

## Quick Start

### 1. Download Datasets

```bash
source venv/bin/activate
python evaluation/datasets/download_datasets.py
```

This downloads:
- **ANLI** (300 samples): Adversarial NLI for contradiction detection
- **VitaminC** (100 samples): Fact verification
- **SciTail** (100 samples): Science-domain NLI
- **SNLI** (100 samples): Classic NLI dataset

**Total: 600 test examples**

### 2. Run Model Comparison

```bash
# Make sure API keys are set
export OPENAI_API_KEY='your-openai-key'
export ANTHROPIC_API_KEY='your-anthropic-key'

# Run comparison
python evaluation/runners/model_comparison.py
```

This will:
1. Test both GPT-4 and Claude Sonnet on contradiction detection
2. Calculate accuracy, precision, recall, F1, calibration error
3. Compare performance by label type (entailment/neutral/contradiction)
4. Measure efficiency (time per example)
5. Generate detailed comparison report

### 3. View Results

Results are saved to `evaluation/reports/comparison_TIMESTAMP.json`

## Available Datasets

| Dataset | Size | Task | Description |
|---------|------|------|-------------|
| ANLI | 300 | Contradiction Detection | Adversarial NLI with entailment/neutral/contradiction labels |
| VitaminC | 100 | Fact Verification | Claims with supporting/refuting evidence |
| SciTail | 100 | Scientific NLI | Entailment/neutral classification |
| SNLI | 100 | Classic NLI | General-purpose natural language inference |

## Evaluation Metrics

### Accuracy Metrics
- **Accuracy**: Overall correctness rate
- **Precision**: Of detected contradictions, how many were real?
- **Recall**: Of real contradictions, how many were detected?
- **F1 Score**: Harmonic mean of precision and recall

### Calibration Metrics
- **Expected Calibration Error (ECE)**: Measures whether confidence scores match actual accuracy
  - Well-calibrated: Claims with 90% confidence are correct 90% of the time
  - Poorly-calibrated: Confidence doesn't match actual performance

### Efficiency Metrics
- **Total Time**: Total execution time
- **Time per Example**: Average processing time per test case

### Label-Specific Metrics
- Accuracy broken down by label type:
  - Entailment detection
  - Neutral detection
  - Contradiction detection

## Architecture

```
evaluation/
├── datasets/
│   ├── download_datasets.py          # Download benchmark datasets
│   ├── anli_samples.json             # 300 ANLI examples
│   ├── vitaminc_samples.json         # 100 fact verification examples
│   └── ...
│
├── benchmarks/
│   └── contradiction_benchmark.py    # ANLI contradiction detection benchmark
│
├── runners/
│   └── model_comparison.py           # Compare multiple LLM providers
│
├── metrics/
│   └── accuracy_metrics.py           # Precision, recall, F1, calibration
│
└── reports/
    └── comparison_YYYYMMDD_HHMMSS.json  # Saved results
```

## Custom Evaluation

### Test a Single Model

```python
from verification_pipeline import AnthropicProvider
from integrated_verification import IntegratedVerificationPipeline
from evaluation.benchmarks.contradiction_benchmark import ContradictionBenchmark

# Setup
provider = AnthropicProvider(api_key="your-key")
pipeline = IntegratedVerificationPipeline(provider)
benchmark = ContradictionBenchmark(pipeline)

# Run on 50 examples
results = benchmark.run_benchmark(limit=50, verbose=True)
```

### Compare Multiple Models

```python
from evaluation.runners.model_comparison import ModelComparison
from verification_pipeline import OpenAIProvider, AnthropicProvider

comparison = ModelComparison()

# Add models
comparison.add_model("GPT-4", OpenAIProvider(api_key=key1, model="gpt-4"))
comparison.add_model("GPT-4-Turbo", OpenAIProvider(api_key=key1, model="gpt-4-turbo"))
comparison.add_model("Claude Sonnet", AnthropicProvider(api_key=key2, model="claude-3-5-sonnet-20241022"))
comparison.add_model("Claude Opus", AnthropicProvider(api_key=key2, model="claude-3-opus-20240229"))

# Run comparison
results = comparison.run_contradiction_benchmark(limit=100)
comparison.print_comparison_report(results)
```

### Use Different Datasets

```python
# Load SNLI instead of ANLI
import json
with open('evaluation/datasets/snli_samples.json') as f:
    snli_data = json.load(f)

# Use VitaminC for fact verification
with open('evaluation/datasets/vitaminc_samples.json') as f:
    vitaminc_data = json.load(f)
```

## Cost Management

**Important**: LLM API calls can be expensive. Start small!

- **Test run**: `limit=10` (10 examples, ~$0.10)
- **Quick eval**: `limit=50` (50 examples, ~$0.50)
- **Full eval**: `limit=100` (100 examples, ~$1.00)
- **Comprehensive**: `limit=300` (all ANLI, ~$3.00)

Costs vary by model:
- Claude Sonnet: ~$0.01 per example
- GPT-4: ~$0.02 per example
- Claude Opus: ~$0.05 per example

## Example Output

```
================================================================================
MODEL COMPARISON REPORT
================================================================================

📊 OVERALL METRICS
--------------------------------------------------------------------------------
Metric               Claude Sonnet          GPT-4
--------------------------------------------------------------------------------
Accuracy                    0.8600 🏆        0.7800
Precision                   0.8421 🏆        0.7692
Recall                      0.8000          0.8333 🏆
F1 Score                    0.8205 🏆        0.8000
Calibration Err             0.0523 🏆        0.0891

⚡ EFFICIENCY METRICS
--------------------------------------------------------------------------------
Metric               Claude Sonnet          GPT-4
--------------------------------------------------------------------------------
Total Time (s)             124.56 🏆       156.23
Time per Example (s)         2.49 🏆         3.12

🏷️  ACCURACY BY LABEL TYPE
--------------------------------------------------------------------------------
Label                Claude Sonnet          GPT-4
--------------------------------------------------------------------------------
contradiction               0.9000 🏆        0.8000
entailment                  0.8500          0.8750 🏆
neutral                     0.8200 🏆        0.6800

💡 SUMMARY
--------------------------------------------------------------------------------
Best Overall (F1): Claude Sonnet (0.8205)
Fastest: Claude Sonnet (2.49s/example)
Best Calibrated: Claude Sonnet (ECE: 0.0523)
```

## Next Steps

1. **Add More Benchmarks**: Create benchmarks for fact verification, logical reasoning
2. **Test More Models**: Add GPT-4-Turbo, Claude Haiku, etc.
3. **Expand Datasets**: Use full ANLI dataset (3000 examples), add LogiQA
4. **Custom Metrics**: Add domain-specific metrics for financial/scientific claims
5. **Human Evaluation**: Manual review of disagreements between models

## Troubleshooting

### Dataset Download Fails
- Some datasets require `trust_remote_code=True`
- Some older datasets use legacy scripts (FEVER, LogiQA)
- Use alternative datasets (VitaminC instead of FEVER, SNLI instead of LogiQA)

### API Rate Limits
- Add delays: `time.sleep(1)` between examples
- Reduce batch size: Use `limit=10` for testing
- Use caching to avoid re-running same examples

### Out of Memory
- Reduce example limit
- Process in batches
- Clear cache between runs

## References

- **ANLI**: Nie et al. (2019). Adversarial NLI: A New Benchmark for Natural Language Understanding
- **FEVER**: Thorne et al. (2018). FEVER: a large-scale dataset for Fact Extraction and VERification
- **SNLI**: Bowman et al. (2015). A large annotated corpus for learning natural language inference
- **VitaminC**: Schuster et al. (2021). Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence
