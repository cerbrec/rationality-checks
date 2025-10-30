# Quick Start Guide

Get started with the core verification pipeline in 5 minutes.

## What is this?

The **Integrated Verification Pipeline** is a hybrid system that catches errors and hallucinations in AI-generated text by combining:
- **Formal verification** for mathematical/quantitative claims (100% certainty)
- **LLM-based verification** for subjective/interpretive claims (confidence scores)

## Installation

```bash
# Clone or navigate to the repository
cd rationality-checks

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your API key (see options below)
```

## API Key Setup

You need **one** of the following:

### Option 1: AWS Bedrock (Recommended)
```bash
# Add to .env:
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=us-east-1
```

### Option 2: Anthropic Claude
```bash
# Add to .env:
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Option 3: OpenAI
```bash
# Add to .env:
OPENAI_API_KEY=sk-...
```

### Option 4: Google Gemini
```bash
# Add to .env:
GEMINI_API_KEY=AIzaSy...
```

## Basic Usage

### Example 1: Verify a Simple Analysis

```python
from src.integrated_verification import IntegratedVerificationPipeline
from src.verification_pipeline import BedrockProvider

# Initialize LLM provider
llm = BedrockProvider(region="us-east-1", model="us.amazon.nova-pro-v1:0")

# Create verification pipeline
pipeline = IntegratedVerificationPipeline(llm)

# Verify an analysis
report = pipeline.verify_analysis(
    original_output="""
    Company X is valued at $50B. Given their $7B in revenue
    and a 10x revenue multiple, this seems reasonable. They
    also have a strong competitive moat in their market.
    """,
    original_query="Analyze Company X's valuation"
)

# Check results
for assessment in report.assessments:
    print(f"\nClaim: {assessment.claim.text}")
    print(f"Confidence: {assessment.overall_confidence:.2f}")
    print(f"Recommendation: {assessment.recommendation}")

    if not all(r.passed for r in assessment.verification_results):
        print("Issues found:")
        for result in assessment.verification_results:
            for issue in result.issues_found:
                print(f"  - {issue}")
```

**Output:**
```
Claim: Company X is valued at $50B
Confidence: 0.00
Recommendation: revise
Issues found:
  - Constraint violated: 50B == 10 * 7B (50B ≠ 70B)
```

### Example 2: Using Different LLM Providers

```python
from src.verification_pipeline import AnthropicProvider, OpenAIProvider, GeminiProvider

# Anthropic Claude
llm = AnthropicProvider(model="claude-3-5-sonnet-20241022")

# OpenAI GPT
llm = OpenAIProvider(model="gpt-4o")

# Google Gemini
llm = GeminiProvider(model="gemini-2.0-flash-exp")

# Use with pipeline
pipeline = IntegratedVerificationPipeline(llm)
```

### Example 3: Command-Line Usage

```bash
# Verify a document from the command line
python examples/verify_document.py document.md

# Specify provider and model
python examples/verify_document.py document.txt \
  --provider bedrock \
  --model nova-pro \
  --output report.json

# See results in report.json
```

## What Gets Verified?

The pipeline checks for:

1. **Mathematical Consistency**
   - Are calculations correct?
   - Do numbers add up?
   - Are formulas consistent?

2. **Logical Consistency**
   - Do claims contradict each other?
   - Are implications sound?
   - Do predictions follow from premises?

3. **Factual Accuracy**
   - Are claims plausible?
   - Do they align with context?
   - Are assumptions reasonable?

4. **Completeness**
   - Are important caveats missing?
   - Is context sufficient?
   - Are edge cases considered?

## Understanding Results

Each claim gets:
- **Confidence score** (0.0 to 1.0)
  - 1.0 = mathematically proven correct
  - 0.0 = mathematically proven wrong or very implausible
  - 0.5-0.8 = uncertain, needs review

- **Recommendation**
  - `keep` - Claim is sound
  - `revise` - Claim has issues, needs correction
  - `flag_uncertainty` - Claim uncertain, add caveats
  - `remove` - Claim likely false

- **Issues** - Specific problems found (if any)

## Next Steps

- **More examples**: See `examples/` directory for detailed use cases
- **Run benchmarks**: `python evaluation/run_comparison.py --limit 10`
- **Full documentation**: See main `README.md`
- **Advanced usage**: Check out `orchestrator/` for multi-step workflows

## Troubleshooting

**"No LLM provider available"**
- Make sure you've added API keys to `.env`
- Check the key is valid and has credits

**"Module not found"**
- Run `pip install -r requirements.txt`
- Make sure you're in the repository root

**Low confidence scores on correct claims**
- Normal! LLM verification is probabilistic
- Confidence < 0.6 triggers review, not automatic rejection
- Mathematical claims get 1.0 or 0.0 (certain)

## Cost Estimates

Typical costs per verification:
- **AWS Bedrock Nova Pro**: ~$0.001 per claim
- **Anthropic Claude**: ~$0.002 per claim
- **OpenAI GPT-4**: ~$0.003 per claim
- **Google Gemini**: ~$0.0005 per claim

A typical document with 10 claims costs **$0.01-0.03** to verify.

---

**Ready to verify?** Try the examples in `examples/basic_usage.py`!
