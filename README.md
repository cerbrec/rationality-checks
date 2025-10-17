# Rationality LLM: Integrated Verification Pipeline

A hybrid verification system that combines formal mathematical verification with LLM-based interpretive verification to ensure accuracy and consistency in AI-generated analysis.

## Overview

When LLMs generate complex analysis (e.g., financial reports, research summaries, technical assessments), they can make subtle errors that are hard to catch:
- Mathematical inconsistencies (e.g., "valued at $50B" with "10x revenue multiple" and "$7B revenue")
- Logical contradictions between claims
- Unsupported interpretive statements
- Missing context or caveats

This pipeline solves these problems by:
1. **Formal verification** for quantitative/logical claims (mathematical certainty)
2. **LLM verification** for interpretive claims (contextual judgment)
3. **Hybrid approach** that uses the right method for each claim type

## Key Features

### 1. Hybrid Verification
- **World State Verification**: Formal mathematical checking for quantitative claims
  - Builds a consistent world model from claims
  - Detects contradictions with mathematical certainty (confidence = 1.0)
  - Validates constraints and equations

- **LLM Verification**: Interpretive checking for subjective claims
  - Empirical testing for logical consistency
  - Fact-checking with external sources
  - Adversarial review to challenge assumptions

### 2. Smart Claim Classification
Automatically categorizes claims as:
- **Formalizable**: Quantitative, factual, causal, logical → World state verification
- **Interpretive**: Subjective analysis, predictions → LLM verification

### 3. Comprehensive Analysis
- Extracts ALL claims from output
- Verifies each claim using appropriate method(s)
- Provides confidence scores and recommendations
- Generates improved output with corrections

### 4. Multi-Model Evaluation
- Built-in evaluation framework with 600+ test cases
- Compare multiple LLM providers (OpenAI, Anthropic, Google, AWS)
- Automated benchmarking on standard datasets (ANLI, VitaminC, SciTail, SNLI)
- Detailed metrics: accuracy, F1, precision, recall, calibration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Original LLM Output                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│           Enhanced Claim Extraction (1 prompt)              │
│  • Extract claims + formal structure                        │
│  • Classify as formalizable or interpretive                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
         ┌─────────┴──────────┐
         │                    │
         ▼                    ▼
┌────────────────┐   ┌────────────────┐
│ Formalizable   │   │ Interpretive   │
│    Claims      │   │    Claims      │
└───────┬────────┘   └────────┬───────┘
        │                     │
        ▼                     │
┌────────────────┐            │
│ World State    │            │
│ Verification   │            │
│ (0 prompts)    │            │
│ • Math proof   │            │
│ • Conf = 1.0   │            │
└───────┬────────┘            │
        │                     │
        └─────────┬───────────┘
                  │
                  ▼
         ┌────────────────┐
         │ LLM Empirical  │
         │     Testing    │
         │   (1 prompt)   │
         │   ALL CLAIMS   │◄─── Both formalizable & interpretive
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  Fact Check    │
         │  (1 prompt)    │
         │   ALL CLAIMS   │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  Adversarial   │
         │    Review      │
         │  (1 prompt)    │
         │   ALL CLAIMS   │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  Completeness  │
         │     Check      │
         │  (1 prompt)    │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │   Synthesis    │
         │  (1 prompt)    │
         └────────┬───────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Verification Report                            │
│  • All claims + assessments                                 │
│  • Confidence scores                                        │
│  • Recommendations (keep/revise/remove)                     │
│  • Improved output                                          │
└─────────────────────────────────────────────────────────────┘
```

**Total: 7 prompts** (same as pure LLM approach, but with mathematical certainty for formal claims)

**Key Points:**
- **Formalizable claims**: Get World State Verification (0 prompts) + all LLM verification methods
- **Interpretive claims**: Skip World State, but still get all LLM verification methods
- **After World State**: Both claim types merge and go through the same pipeline

## Installation

### Requirements
- Python 3.10 or higher
- pip
- API key for at least one LLM provider (see Configuration below)

### Setup

```bash
# Clone the repository
git clone https://github.com/drwiner/rationality-llm.git
cd rationality-llm

# Run the setup script (recommended)
./setup.sh

# Or manually install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys
```

### Environment Configuration

Copy `.env.example` to `.env` and configure your chosen provider(s):

```bash
# Anthropic Claude (recommended)
ANTHROPIC_API_KEY=sk-ant-api03-...

# OpenAI GPT
OPENAI_API_KEY=sk-...

# Google Gemini
GEMINI_API_KEY=AIzaSy...

# AWS Bedrock (optional)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

See [.env.example](.env.example) for complete configuration options.

## Quick Start

### Basic Usage

```python
from integrated_verification import IntegratedVerificationPipeline
from verification_pipeline import BedrockProvider

# Initialize with Amazon Nova Pro (recommended for best performance)
llm = BedrockProvider(
    region="us-east-1",
    model="us.amazon.nova-pro-v1:0"
)
pipeline = IntegratedVerificationPipeline(llm)

# Alternative: Use Anthropic Claude
# from verification_pipeline import AnthropicProvider
# llm = AnthropicProvider(api_key="your-api-key")

# Verify an analysis
report = pipeline.verify_analysis(
    original_output="""
    Company X is valued at $50B. Given their $7B in revenue
    and a 10x revenue multiple, this seems reasonable. They
    also have a strong competitive moat in their market.
    """,
    original_query="Analyze Company X's valuation"
)

# Review results
for assessment in report.assessments:
    print(f"\nClaim: {assessment.claim.text}")
    print(f"Confidence: {assessment.overall_confidence:.2f}")
    print(f"Recommendation: {assessment.recommendation}")

    if not all(r.passed for r in assessment.verification_results):
        print(f"Issues found:")
        for result in assessment.verification_results:
            for issue in result.issues_found:
                print(f"  - {issue}")
```

### Example Output

```
Claim: Company X is valued at $50B
Confidence: 0.00
Recommendation: revise
Issues found:
  - Constraint violated: 50000000000 == 10 * 7000000000 (50B ≠ 70B)

Claim: Has strong competitive moat
Confidence: 0.70
Recommendation: flag_uncertainty
```

### Command-Line Model Comparison

Compare multiple LLM providers on contradiction detection:

```bash
# Quick test with Amazon Nova Pro (recommended)
python run_comparison.py --limit 10 --models bedrock-nova-pro

# Compare Nova Pro vs others
python run_comparison.py --limit 20 --models bedrock-nova-pro claude gpt4o

# Full comparison with all models
python run_comparison.py --limit 20 --models bedrock-nova-pro gpt4 gpt4o claude gemini
```

See [CLI Tools](#cli-tools) for more details.

## LLM Providers

The pipeline supports multiple LLM providers. Configure via `.env` file or pass API keys directly.

### Anthropic Claude

```python
from verification_pipeline import AnthropicProvider

llm = AnthropicProvider(
    api_key="your-api-key",
    model="claude-3-5-sonnet-20241022"  # Latest Sonnet (recommended)
    # model="claude-3-opus-20240229"    # Most capable
    # model="claude-3-haiku-20240307"   # Fastest, cheapest
)
```

**Setup**: Set `ANTHROPIC_API_KEY` in `.env`
**Get API Key**: https://console.anthropic.com/

### OpenAI GPT

```python
from verification_pipeline import OpenAIProvider

llm = OpenAIProvider(
    api_key="your-api-key",
    model="gpt-4o"         # Latest (recommended)
    # model="gpt-4"        # Original GPT-4
    # model="gpt-4-turbo"  # Fast GPT-4
)
```

**Setup**: Set `OPENAI_API_KEY` in `.env`
**Get API Key**: https://platform.openai.com/api-keys

### Google Gemini

```python
from verification_pipeline import GeminiProvider

llm = GeminiProvider(
    api_key="your-api-key",
    model="gemini-2.0-flash-exp"  # Fast (default)
    # model="gemini-2.5-pro-002"  # Most capable
)
```

**Setup**: Set `GEMINI_API_KEY` in `.env`
**Get API Key**: https://aistudio.google.com/app/apikey
**Documentation**: See [GEMINI_SETUP.md](GEMINI_SETUP.md)

### AWS Bedrock

```python
from verification_pipeline import BedrockProvider

llm = BedrockProvider(
    region="us-east-1",
    model="us.amazon.nova-pro-v1:0"                      # Amazon Nova Pro (recommended)
    # model="us.anthropic.claude-sonnet-4-20250514-v1:0"  # Claude Sonnet 4
    # model="us.anthropic.claude-opus-4-1-20250805-v1:0"   # Claude Opus 4
    # model="us.amazon.nova-premier-v1:0"                  # Amazon Nova Premier
)
```

**Setup**: Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` in `.env`
**Requirements**: AWS account with Bedrock access
**Recommended**: Amazon Nova Pro offers excellent accuracy (60-82% on benchmarks) with fast processing (8-14s per example)

### Custom Provider

```python
from verification_pipeline import LLMProvider

class CustomProvider(LLMProvider):
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Your implementation
        return response
```

### Provider Comparison

| Provider | Models | Speed | Cost | Accuracy (ANLI)* | Setup Difficulty | Recommended |
|----------|--------|-------|------|-----------------|------------------|-------------|
| **AWS Bedrock** | Claude, Nova, Llama | Fast | $-$$ | **60-82%** | Medium | ✅ **Nova Pro** |
| **Anthropic** | Claude 3.5, Opus, Haiku | Fast | $$ | 48-56% | Easy | Claude 4.5 |
| **OpenAI** | GPT-4o, GPT-4, Turbo | Fast | $$ | TBD | Easy | GPT-4o |
| **Google** | Gemini 2.0/2.5 | Very Fast | $ | TBD | Easy | Gemini 2.0 |

*Accuracy on ANLI contradiction detection benchmark (50 examples). See [Evaluation Results](#evaluation-results) below.

See [MODELS_REFERENCE.md](MODELS_REFERENCE.md) for complete model listing and cost comparison.

## Examples

The repository includes comprehensive examples in `examples/`:

### 1. Basic Usage (`examples/basic_usage.py`)

Four complete examples demonstrating different use cases:

```bash
python examples/basic_usage.py
```

**Includes:**
- **Financial Analysis**: Detecting mathematical contradictions in valuations
- **Research Verification**: Checking sample size consistency and statistical claims
- **Technical Specifications**: Validating performance metrics and calculations
- **Custom Workflow**: Extending the pipeline for specific needs

### 2. Interactive Testing (`test_interactive_verification.py`)

Step-by-step walkthrough of the verification pipeline:

```bash
# Interactive mode (press Enter to continue between steps)
python test_interactive_verification.py

# Automated mode (no pauses)
python test_interactive_verification.py --auto
```

**Shows:**
- Claim extraction with formal structures
- World state construction step-by-step
- Consistency analysis in real-time
- Complete verification breakdown

### 3. Athlete Report Analysis (`examples/athlete_report_test.py`)

Real-world example analyzing complex athlete performance reports:

```bash
# Show expected findings
python examples/athlete_report_test.py --expected

# Run with Claude
python examples/athlete_report_test.py --provider=anthropic

# Run with GPT-4
python examples/athlete_report_test.py --provider=openai
```

**Documentation**: See [TESTING.md](TESTING.md) for detailed usage guide.

## Evaluation Framework

The project includes a comprehensive evaluation system for comparing LLM performance on contradiction detection.

### Datasets

Built-in evaluation on 600+ examples across 4 datasets:
- **ANLI** (300): Adversarial NLI for contradiction detection
- **VitaminC** (100): Fact verification
- **SciTail** (100): Scientific NLI
- **SNLI** (100): Classic NLI benchmark

### Running Evaluations

```bash
# Quick test (10 examples, ~$0.10)
python run_comparison.py --limit 10 --models gpt4 claude

# Medium evaluation (50 examples, ~$0.50)
python run_comparison.py --limit 50 --models gpt4o claude gemini

# Full evaluation (100 examples, ~$1.00)
python run_comparison.py --limit 100 --models gpt4o claude-opus gemini-pro
```

### Metrics Measured

For each model:
- **Accuracy**: Overall correctness
- **F1 Score**: Balance of precision and recall
- **Precision**: Of flagged issues, how many are real?
- **Recall**: Of real issues, how many were caught?
- **Calibration**: Does confidence match actual performance?
- **Speed**: Processing time per example

### Example Results

```
📊 OVERALL METRICS (ANLI Dataset)
--------------------------------------------------------------------------------
Metric               Nova Pro 🏆        Claude 3.7         GPT-4o
--------------------------------------------------------------------------------
Accuracy                0.60 🏆            0.48              TBD
F1 Score                0.52 🏆            0.46              TBD
Precision               0.41               0.33              TBD
Recall                  0.73               0.73              TBD
Speed (s/example)       13.9 🏆            27.1              TBD
```

**Latest Results (50 examples, ANLI + SCITAIL):**
- **Amazon Nova Pro**: 60% accuracy (ANLI), 82% accuracy (SCITAIL), ~2x faster than Claude
- **Claude 3.7 Sonnet**: 48% accuracy (ANLI), 66% accuracy (SCITAIL)

Results are saved to `evaluation/reports/comparison_TIMESTAMP.json`.

**Documentation**: See [EVALUATION_SETUP.md](EVALUATION_SETUP.md) for complete evaluation guide.

## Evaluation Results

Based on standardized contradiction detection benchmarks (50 examples per dataset):

### Amazon Nova Pro (AWS Bedrock) - **Recommended**

**ANLI (Adversarial NLI):**
- Accuracy: 60%
- F1 Score: 0.52
- Precision: 40.7%
- Recall: 73.3%
- Speed: 13.9s per example

**SCITAIL (Scientific NLI):**
- Accuracy: 82%
- Speed: 8.3s per example
- Excellent performance on scientific/technical content

**Key Strengths:**
- ✅ Best overall accuracy across tested datasets
- ✅ 2x faster than Claude 3.7 (8-14s vs 16-27s per example)
- ✅ Excellent recall (73%) - catches most contradictions
- ✅ Cost-effective via AWS Bedrock
- ⚠️ Note: May over-flag neutral statements as contradictions (lower precision)

### Claude 3.7 Sonnet (Anthropic)

**ANLI (Adversarial NLI):**
- Accuracy: 48%
- F1 Score: 0.46
- Speed: 27.1s per example

**SNLI (Stanford NLI):**
- Accuracy: 56%
- F1 Score: 0.48
- Speed: 15.7s per example

**SCITAIL (Scientific NLI):**
- Accuracy: 66%
- Speed: 16.2s per example

**VITAMINC (Fact Verification):**
- Accuracy: 66%
- F1 Score: 0.45
- Speed: 20.1s per example

**Key Characteristics:**
- More conservative in flagging contradictions
- Tested across all 4 datasets
- Slower processing speed
- Lower accuracy compared to Nova Pro

### Model Recommendations by Use Case

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Production deployment** | Amazon Nova Pro | Best accuracy + speed + cost ratio |
| **High recall needed** | Amazon Nova Pro | 73% recall, catches most issues |
| **High precision needed** | Claude 4.5 Sonnet | Fewer false positives |
| **Budget-conscious** | Gemini 2.0 Flash | Lowest cost, good speed |
| **Maximum capability** | GPT-4o or Claude Opus | Latest models (benchmarks pending) |

**Next Steps:** Run evaluation on GPT-4o, Claude 4.5 Sonnet, and Gemini 2.0 Flash for complete comparison.

## CLI Tools

### `run_comparison.py`

Compare multiple LLM providers on contradiction detection benchmarks.

```bash
# Basic usage
python run_comparison.py --limit 20 --models gpt4 claude

# Options
python run_comparison.py \
  --limit 50 \                    # Number of test examples
  --models gpt4o claude gemini \  # Models to compare
  --datasets anli snli \          # Datasets to use (default: all)
  --verbose                       # Show detailed progress
```

**Available Models:**
- OpenAI: `gpt4`, `gpt4o`, `gpt4-turbo`
- Anthropic: `claude`, `claude-opus`, `claude-haiku`
- Google: `gemini`, `gemini-pro`
- AWS Bedrock: `bedrock-claude`, `bedrock-nova-pro`, `bedrock-llama-3.3-70b`

**Available Datasets:**
- `anli`: Adversarial NLI (300 examples)
- `snli`: Stanford NLI (100 examples)
- `scitail`: Scientific NLI (100 examples)
- `vitaminc`: Fact verification (100 examples)

### Output

Results are automatically saved to:
- JSON: `evaluation/reports/comparison_TIMESTAMP.json`
- Console: Formatted comparison tables

## Interactive Testing

### Step-by-Step Verification

Walk through the verification pipeline interactively:

```bash
python test_interactive_verification.py
```

**Features:**
- Pause between each step to review output
- Detailed world state construction visualization
- Claim-by-claim breakdown
- Consistency analysis with explanations

### Viewing Modes

```bash
# Interactive mode (default)
python test_interactive_verification.py

# Automated mode (no pauses)
python test_interactive_verification.py --auto

# Custom report
python examples/athlete_report_test.py --report=/path/to/report.json
```

**Documentation**: See [TESTING.md](TESTING.md) for complete testing guide.

## Use Cases

### 1. Financial Analysis Verification
```python
# Verify financial models for mathematical consistency
report = pipeline.verify_analysis(
    original_output="Company valuation report with P/E ratios, revenue multiples...",
    original_query="Analyze company financials"
)
```

### 2. Research Paper Review
```python
# Check research claims and logical consistency
report = pipeline.verify_analysis(
    original_output="Research paper with hypotheses, data, conclusions...",
    original_query="Review research methodology"
)
```

### 3. Technical Documentation
```python
# Verify technical claims and specifications
report = pipeline.verify_analysis(
    original_output="System architecture with performance metrics...",
    original_query="Document system capabilities"
)
```

## Comparison: Original vs Integrated Pipeline

### Example: Financial Analysis

**Claims:**
1. "Company X is valued at $50B" (quantitative)
2. "Uses 10x revenue multiple" (factual)
3. "Revenue is $7B" (quantitative)
4. "Has strong competitive moat" (interpretive)

### Original LLM-Only Pipeline
- **Method**: LLM empirical testing for all claims
- **Result**: Might not catch the mathematical contradiction
- **Confidence**: 0.7-0.8 (LLM judgment)

### Integrated Pipeline
- **World State Verification** (Claims 1-3):
  - Builds state: `{valuation: 50B, multiple: 10, revenue: 7B}`
  - Checks constraint: `50B == 10 * 7B?`
  - Result: **VIOLATED** (50B ≠ 70B)
  - **Confidence: 1.0** (mathematical proof)

- **LLM Empirical Test** (Claim 4):
  - Evaluates interpretive claim
  - Confidence: 0.7 (contextual judgment)

### Key Advantages
1. **Mathematical certainty** for quantitative claims
2. **Cross-claim contradiction detection**
3. **Same prompt count** (7 total)
4. **Right tool for each claim type**

## Advanced Usage

### Custom Verification Thresholds

```python
# Adjust confidence thresholds for recommendations
def custom_recommendation(claim, results, confidence):
    if confidence < 0.3:
        return "remove", None
    elif confidence < 0.8:
        return "flag_uncertainty", f"{claim.text} [Needs verification]"
    else:
        return "keep", None

# Use custom logic
pipeline._make_recommendation = custom_recommendation
```

### Accessing World State

```python
# Get the world state after verification
formal_claims = [c for c in claims if c.is_formalizable]
world_results, world_state = pipeline._world_state_verify(formal_claims)

# Query the world state
company_props = world_state.query("Company_X")
for prop in company_props:
    print(f"{prop.predicate}: {prop.value}")
```

### Batch Processing

```python
# Verify multiple outputs
outputs = [
    ("Output 1 text...", "Query 1"),
    ("Output 2 text...", "Query 2"),
    # ...
]

reports = [
    pipeline.verify_analysis(output, query)
    for output, query in outputs
]
```

### Custom Evaluation Benchmarks

```python
from evaluation.benchmarks.contradiction_benchmark import ContradictionBenchmark
from evaluation.metrics.accuracy_metrics import calculate_accuracy_metrics

# Create custom test cases
custom_cases = [
    {
        'premise': 'Company valued at $50B with 10x revenue multiple',
        'hypothesis': 'Company has $5B in revenue',
        'label': 0  # entailment
    },
    {
        'premise': 'Company valued at $50B with 10x revenue multiple',
        'hypothesis': 'Company has $7B in revenue',
        'label': 2  # contradiction
    }
]

# Run benchmark
benchmark = ContradictionBenchmark(pipeline)
results = benchmark.run_on_dataset(custom_cases, verbose=True)

# Calculate metrics
metrics = calculate_accuracy_metrics(
    predictions=[r['predicted_label'] for r in results],
    ground_truth=[r['true_label'] for r in results],
    confidences=[r['confidence'] for r in results]
)

print(f"Accuracy: {metrics['accuracy']:.2f}")
print(f"F1 Score: {metrics['f1_score']:.2f}")
```

## Architecture Details

### Claim Types

The system recognizes and handles different claim types:

- **FACTUAL**: Verifiable facts (e.g., "The company was founded in 2010")
- **QUANTITATIVE**: Numerical claims (e.g., "Revenue is $100M")
- **CAUSAL**: Cause-effect (e.g., "Price increase led to demand drop")
- **LOGICAL**: Logical inferences (e.g., "If A then B")
- **INTERPRETIVE**: Subjective analysis (e.g., "Strong market position")
- **PREDICTIVE**: Future predictions (e.g., "Will grow 20% next year")
- **ASSUMPTION**: Stated/implicit assumptions (e.g., "Assuming constant growth")

### Verification Methods

Each claim is verified using one or more methods:

1. **World State Verification** (formal claims)
   - Mathematical proof
   - Constraint checking
   - Contradiction detection

2. **Empirical Testing** (interpretive claims)
   - State transition tests
   - Contradiction tests
   - Testable predictions

3. **Fact Checking** (factual claims)
   - External source verification
   - Cross-referencing

4. **Adversarial Review** (all claims)
   - Challenge assumptions
   - Find edge cases
   - Alternative interpretations

## Performance

### Prompt Efficiency
- **7 total prompts** for complete verification
- World state verification adds **0 prompts** (pure computation)
- Batch processing for efficiency

### Accuracy Improvements
- **Mathematical certainty** (confidence = 1.0) for formal claims
- Catches contradictions that LLMs miss
- Better calibrated confidence scores

### Speed Benchmarks

Based on evaluation results (contradiction detection with verification pipeline):

| Provider | Model | Speed (s/example) | Cost/1000 examples | Accuracy (ANLI) |
|----------|-------|-------------------|-------------------|-----------------|
| AWS Bedrock | **Amazon Nova Pro** | **8.3-13.9** | **$3-5** | **60%** 🏆 |
| Google | Gemini 2.0 Flash | 3.9* | $2 | TBD |
| Anthropic | Claude Haiku | 4.8* | $3 | TBD |
| OpenAI | GPT-4o | 5.2* | $10 | TBD |
| Anthropic | Claude Sonnet | 6.1* | $10 | TBD |
| Anthropic | Claude 3.7 Sonnet | 16.2-27.1 | $15-25 | 48% |
| OpenAI | GPT-4 | 7.3* | $20 | TBD |
| Anthropic | Claude Opus | 9.4* | $30 | TBD |

*Estimated based on typical API response times. Amazon Nova Pro and Claude 3.7 measured on actual evaluation runs.

## Limitations

1. **Formal Structure Extraction**: Requires LLM to accurately extract formal structure from claims
2. **Simple Constraint Solving**: Currently uses basic equation solving (can be extended with SMT solvers)
3. **Natural Language Understanding**: Interpretive verification still relies on LLM capabilities
4. **Computational Cost**: 7 LLM calls per verification (can be expensive for large documents)
5. **Single Document Focus**: Currently optimized for individual document verification

## Roadmap

- [ ] Enhanced constraint solver (SMT solver integration with Z3)
- [ ] Support for temporal reasoning
- [ ] Multi-document verification and cross-document consistency
- [ ] Interactive correction mode with user feedback
- [ ] Performance optimization (caching, parallel processing)
- [ ] Web interface for visualization
- [ ] Integration with popular LLM frameworks (LangChain, LlamaIndex)
- [ ] Support for more datasets and benchmarks
- [ ] Confidence calibration improvements
- [ ] Domain-specific verification modules

## Additional Documentation

This README provides an overview. For detailed information, see:

- **[TESTING.md](TESTING.md)**: Interactive testing guide, athlete report examples
- **[EVALUATION_SETUP.md](EVALUATION_SETUP.md)**: Complete evaluation framework guide
- **[GEMINI_SETUP.md](GEMINI_SETUP.md)**: Google Gemini integration setup
- **[MODELS_REFERENCE.md](MODELS_REFERENCE.md)**: Complete model listing and comparison
- **[PYDANTIC_VALIDATION.md](PYDANTIC_VALIDATION.md)**: Data validation details
- **[.env.example](.env.example)**: Environment configuration template

## Project Structure

```
rationality-llm/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.sh                           # Setup script
├── .env.example                       # Environment template
│
├── integrated_verification.py         # Main pipeline (838 lines)
├── verification_pipeline.py           # LLM providers (424 lines)
├── world_state_verification.py        # Formal verification (415 lines)
├── run_comparison.py                  # CLI comparison tool (240 lines)
│
├── examples/
│   ├── basic_usage.py                # 4 complete examples
│   └── athlete_report_test.py        # Real-world testing
│
├── evaluation/
│   ├── datasets/                     # Test datasets (600+ examples)
│   │   ├── anli_samples.json
│   │   ├── snli_samples.json
│   │   ├── scitail_samples.json
│   │   └── vitaminc_samples.json
│   ├── benchmarks/                   # Benchmark implementations
│   │   └── contradiction_benchmark.py
│   ├── metrics/                      # Evaluation metrics
│   │   └── accuracy_metrics.py
│   ├── runners/                      # Comparison runners
│   │   └── model_comparison.py
│   └── reports/                      # Saved results
│
├── docs/                             # Additional documentation
├── test_interactive_verification.py  # Interactive testing (450 lines)
└── test_bedrock.py                   # AWS Bedrock testing
```

## Contributing

Contributions are welcome! Areas for improvement:

1. **Better formal structure extraction** - Improve claim parsing and formalization
2. **More sophisticated constraint solving** - Integrate SMT solvers (Z3, CVC5)
3. **Additional verification methods** - Temporal reasoning, probabilistic claims
4. **Performance optimizations** - Caching, parallel processing, batching
5. **Testing and examples** - More real-world examples and edge cases
6. **Documentation** - Tutorials, videos, case studies
7. **Integrations** - LangChain, LlamaIndex, other frameworks

Please submit issues and pull requests on [GitHub](https://github.com/drwiner/rationality-llm).

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this in your research, please cite:

```bibtex
@software{rationality_llm,
  title={Rationality LLM: Integrated Verification Pipeline},
  author={David R. Winer},
  year={2025},
  url={https://github.com/drwiner/rationality-llm}
}
```

## Contact

- **GitHub Issues**: [github.com/drwiner/rationality-llm/issues](https://github.com/drwiner/rationality-llm/issues)
- **Email**: drwiner131 at gmail.com

## Acknowledgments

This project combines ideas from:
- Formal verification and constraint solving
- LLM-based fact checking and verification
- Adversarial testing and red-teaming
- Epistemic rationality and claim assessment
- Research on LLM calibration and confidence estimation
