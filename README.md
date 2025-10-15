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
        ▼                     ▼
┌────────────────┐   ┌────────────────┐
│ World State    │   │ LLM Empirical  │
│ Verification   │   │     Testing    │
│ (0 prompts)    │   │   (1 prompt)   │
│ • Math proof   │   │ • Logic check  │
│ • Conf = 1.0   │   │ • Conf = 0-1   │
└───────┬────────┘   └────────┬───────┘
        │                     │
        └─────────┬───────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Fact Check    │
         │  (1 prompt)    │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  Adversarial   │
         │    Review      │
         │  (1 prompt)    │
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

## Installation

### Requirements
- Python 3.10 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rationality-llm.git
cd rationality-llm

# Run the setup script (recommended)
./setup.sh

# Or manually install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from integrated_verification import IntegratedVerificationPipeline
from verification_pipeline import AnthropicProvider

# Initialize with your LLM provider
llm = AnthropicProvider(api_key="your-api-key")
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

## Configuration

### LLM Providers

The pipeline supports multiple LLM providers:

#### Anthropic Claude
```python
from verification_pipeline import AnthropicProvider

llm = AnthropicProvider(
    api_key="your-api-key",
    model="claude-3-5-sonnet-20241022"  # or other models
)
```

#### OpenAI GPT
```python
from verification_pipeline import OpenAIProvider

llm = OpenAIProvider(
    api_key="your-api-key",
    model="gpt-4"  # or "gpt-4-turbo", "gpt-3.5-turbo"
)
```

#### Custom Provider
```python
from verification_pipeline import LLMProvider

class CustomProvider(LLMProvider):
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Your implementation
        return response
```

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

## Limitations

1. **Formal Structure Extraction**: Requires LLM to accurately extract formal structure from claims
2. **Simple Constraint Solving**: Currently uses basic equation solving (can be extended)
3. **Natural Language Understanding**: Interpretive verification still relies on LLM capabilities
4. **Computational Cost**: 7 LLM calls per verification (can be expensive for large documents)

## Roadmap

- [ ] Enhanced constraint solver (SMT solver integration)
- [ ] Support for temporal reasoning
- [ ] Multi-document verification
- [ ] Interactive correction mode
- [ ] Performance optimization (caching, parallel processing)
- [ ] Web interface for visualization
- [ ] Integration with popular LLM frameworks (LangChain, LlamaIndex)

## Contributing

Contributions are welcome! Areas for improvement:

1. **Better formal structure extraction**
2. **More sophisticated constraint solving**
3. **Additional verification methods**
4. **Performance optimizations**
5. **Testing and examples**

## License

MIT License - see LICENSE file for details

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

- Issues: [GitHub Issues](https://github.com/drwiner/rationality-llm/issues)
- Email: drwiner131 at gmail.com

## Acknowledgments

This project combines ideas from:
- Formal verification and constraint solving
- LLM-based fact checking and verification
- Adversarial testing and red-teaming
- Epistemic rationality and claim assessment
