# Evaluation Framework Setup Complete ✅

A comprehensive evaluation framework has been created to compare OpenAI GPT-4 vs Claude on your rationality-llm tasks.

## What Was Built

### 1. Dataset Collection (600 samples)
- ✅ **ANLI** (300 samples): Adversarial NLI for contradiction detection - **Most important for your use case**
- ✅ **VitaminC** (100 samples): Fact verification dataset
- ✅ **SciTail** (100 samples): Scientific NLI
- ✅ **SNLI** (100 samples): Classic NLI benchmark

### 2. Evaluation Infrastructure
- ✅ **Metrics Module** (`evaluation/metrics/accuracy_metrics.py`)
  - Accuracy, Precision, Recall, F1 Score
  - Confidence calibration (Expected Calibration Error)
  - Confusion matrices

- ✅ **Contradiction Benchmark** (`evaluation/benchmarks/contradiction_benchmark.py`)
  - Tests pipeline on ANLI dataset
  - Measures contradiction detection accuracy
  - Breaks down performance by label type

- ✅ **Model Comparison Framework** (`evaluation/runners/model_comparison.py`)
  - Side-by-side comparison of multiple LLMs
  - Unified metrics across all models
  - Generates detailed comparison reports

### 3. Documentation
- ✅ Comprehensive README in `evaluation/README.md`
- ✅ Usage examples and troubleshooting

## Quick Start Guide

### Step 1: Verify Setup

```bash
cd /Users/drw/cerbrec/rationality-llm
source venv/bin/activate

# Verify datasets are downloaded
ls evaluation/datasets/*.json
# Should show: anli_samples.json, vitaminc_samples.json, scitail_samples.json, snli_samples.json
```

### Step 2: Set API Keys

```bash
# Edit your .env file or export directly
export OPENAI_API_KEY='your-openai-key-here'
export ANTHROPIC_API_KEY='your-anthropic-key-here'
```

### Step 3: Run Your First Comparison (Small Test)

```bash
# Test on just 10 examples first (~$0.10 cost)
python evaluation/runners/model_comparison.py
```

**Note**: The script is currently configured to run on 20 examples. Edit line 238 in `model_comparison.py` to change:
```python
results = comparison.run_contradiction_benchmark(limit=10, verbose=False)  # Start with 10
```

### Step 4: Run Full Evaluation

Once you verify it works, increase the sample size:

```python
# In model_comparison.py, line 238:
results = comparison.run_contradiction_benchmark(limit=100, verbose=False)  # Full eval
```

## What Gets Measured

### For Each Model:
1. **Contradiction Detection Accuracy**
   - Can it identify when hypothesis contradicts premise?
   - Precision: Of flagged contradictions, how many were real?
   - Recall: Of real contradictions, how many were caught?

2. **Performance by Claim Type**
   - Entailment detection (premise → hypothesis is true)
   - Neutral detection (independent claims)
   - Contradiction detection (premise contradicts hypothesis)

3. **Confidence Calibration**
   - Does confidence match actual correctness?
   - If model is 90% confident, is it right 90% of the time?

4. **Efficiency**
   - Time per example
   - Total processing time

## Understanding the Results

### Example Output:
```
📊 OVERALL METRICS
--------------------------------------------------------------------------------
Metric               Claude Sonnet          GPT-4
--------------------------------------------------------------------------------
Accuracy                    0.8600 🏆        0.7800
F1 Score                    0.8205 🏆        0.8000
```

**What this means:**
- Claude Sonnet correctly identified contradictions 86% of the time
- GPT-4 was correct 78% of the time
- 🏆 marks the winner for each metric

### Key Metrics to Watch:

1. **F1 Score** (Most Important)
   - Balances precision and recall
   - Best single metric for contradiction detection

2. **Recall** (Critical for Your Use Case)
   - % of real contradictions that were caught
   - High recall = fewer missed errors
   - Low recall = dangerous (misses real problems)

3. **Calibration Error** (Lower is Better)
   - Measures confidence accuracy
   - Low ECE = trustworthy confidence scores
   - High ECE = model overconfident or underconfident

## Cost Estimates

| Test Size | Examples | Approx. Cost | Use Case |
|-----------|----------|--------------|----------|
| Tiny | 10 | $0.10 | Quick test |
| Small | 20 | $0.20 | Initial comparison |
| Medium | 50 | $0.50 | Solid eval |
| Large | 100 | $1.00 | Comprehensive |
| Full ANLI | 300 | $3.00 | Complete benchmark |

*Costs assume ~$0.01 per example (varies by model and prompt length)*

## Next Steps

### Option 1: Test Different Models
```python
# In model_comparison.py, add more models:
comparison.add_model("GPT-4-Turbo", OpenAIProvider(api_key=key, model="gpt-4-turbo"))
comparison.add_model("Claude Opus", AnthropicProvider(api_key=key, model="claude-3-opus-20240229"))
comparison.add_model("Claude Haiku", AnthropicProvider(api_key=key, model="claude-3-haiku-20240307"))
```

### Option 2: Test Different Pipeline Configurations
Compare your integrated pipeline (with world state verification) vs pure LLM:

```python
# Test with world state verification
pipeline_integrated = IntegratedVerificationPipeline(provider)

# vs pure LLM (disable world state)
# (You'd need to add a flag to disable formal verification)
```

### Option 3: Add More Benchmarks
Create new benchmarks for:
- Fact verification (using VitaminC)
- Claim extraction accuracy
- Different types of logical reasoning

### Option 4: Test on Your Own Data
```python
# Create custom test cases from your domain
custom_examples = [
    {
        'premise': 'Company valued at $50B with 10x revenue multiple',
        'hypothesis': 'Company has $5B in revenue',
        'label': 0  # entailment
    },
    {
        'premise': 'Company valued at $50B with 10x revenue multiple',
        'hypothesis': 'Company has $7B in revenue',
        'label': 2  # contradiction (should be $5B)
    }
]
```

## File Structure

```
evaluation/
├── README.md                          # Detailed documentation
├── __init__.py
├── datasets/
│   ├── download_datasets.py           # Download ANLI, FEVER, etc.
│   ├── download_alternative_datasets.py
│   ├── anli_samples.json              # 300 contradiction examples
│   ├── vitaminc_samples.json          # 100 fact verification
│   ├── scitail_samples.json           # 100 NLI examples
│   └── snli_samples.json              # 100 NLI examples
├── benchmarks/
│   └── contradiction_benchmark.py     # Test on ANLI
├── runners/
│   └── model_comparison.py            # Compare multiple models
├── metrics/
│   └── accuracy_metrics.py            # All evaluation metrics
└── reports/
    └── comparison_TIMESTAMP.json      # Saved results
```

## Troubleshooting

### Issue: "No module named 'evaluation'"
```bash
# Make sure you're in the project root
cd /Users/drw/cerbrec/rationality-llm
python evaluation/runners/model_comparison.py
```

### Issue: API Key Not Found
```bash
# Check if keys are loaded
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"

# If not, load from .env
source venv/bin/activate
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"
```

### Issue: Too Expensive
- Start with `limit=10`
- Use cheaper models (Claude Haiku instead of Opus)
- Cache results to avoid re-running

### Issue: Too Slow
- Reduce sample size
- Run on smaller dataset
- Add parallel processing (requires code changes)

## Advanced Usage

### Save and Compare Historical Results
```bash
# Results are auto-saved with timestamps
ls evaluation/reports/

# Compare different runs
python -c "
import json
with open('evaluation/reports/comparison_20250114_120000.json') as f:
    run1 = json.load(f)
# ... compare runs
"
```

### Custom Metrics
Add your own metrics in `evaluation/metrics/accuracy_metrics.py`:
```python
def calculate_domain_specific_metric(predictions, ground_truth):
    # Your custom logic
    return metric_value
```

## Key Insights from Research

Based on the dataset research, here's what we learned:

1. **ANLI is Perfect for You**
   - Specifically designed to test contradiction detection
   - Adversarial examples that fool most models
   - 3 difficulty rounds (we're using all 3)

2. **Current SOTA Performance**
   - Best models (QwQ-32B): ~93% on ReClor
   - GPT-4: ~66% on logical reasoning
   - Your mileage may vary on ANLI

3. **Calibration Matters**
   - LLMs often overconfident
   - Calibration error tells you if confidence is trustworthy
   - Critical for your use case (flagging uncertain claims)

## Questions?

- Check `evaluation/README.md` for detailed usage
- Look at example in `model_comparison.py` main()
- Run with `verbose=True` to see detailed progress

## Ready to Run!

Everything is set up. Just run:

```bash
cd /Users/drw/cerbrec/rationality-llm
source venv/bin/activate
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'
python evaluation/runners/model_comparison.py
```

Good luck with your evaluation! 🚀
