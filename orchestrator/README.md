# Unified Intelligent Orchestrator

**Status:** Experimental / Research

**Purpose:** Advanced multi-step workflow system that integrates verification skills during generation, not as post-processing.

---

## ⚠️ Important Notice

This orchestrator is an **experimental research system** built on top of the core verification pipeline. If you're looking for the **production-ready verification system**, you want:

👉 **[Core Verification Pipeline](../src/integrated_verification.py)** - See [QUICKSTART.md](../QUICKSTART.md)

---

## What is this?

The Unified Intelligent Orchestrator implements a **7-step workflow** where each step:
1. Generates content using AI
2. Uses verification skills in real-time
3. Only proceeds if verification passes

This is different from the core pipeline which verifies *after* generation.

### Architecture: Generation + Verification

```
Step 1: Goal Template → Domain Detection
Step 2: Resolution Strategy → Verification Strategy Selection
Step 3: Information Collection → Fact Checking (web search + APIs)
Step 4: Data Processing → World State Verification (math checks)
Step 5: Rational Connection → Consistency Checks (empirical testing)
Step 6: AI Prediction → Adversarial Review (challenge claims)
Step 7: Final Artifact → Completeness + Synthesis
```

## Current Implementation Status

### ✅ Working Components

- **7-step workflow orchestration** - Fully functional
- **Verification skills integration** - FactChecking, WorldState, Synthesis working
- **Web search integration** - Serper API connected
- **NIL domain** - College athlete valuation reports
- **Bedrock-first LLM provider** - AWS Bedrock + Anthropic fallback

### ⚠️ Experimental Components

- **EmpiricalTestingSkill** - LLM-based, recently implemented
- **AdversarialReviewSkill** - LLM-based, recently implemented
- **CompletenessCheckSkill** - Placeholder (returns static confidence)

### ❌ Placeholder Components

**Domain-specific skills use fake data:**
- NIL player stats - Returns placeholder values (needs ESPN API)
- NIL market data - Returns fake valuations (needs On3/Opendorse API)
- Team context - Returns unknown values (needs NCAA database)

**See [archived implementation plan](../archive/implementation_notes/IMPLEMENTATION_PLAN.md) for details.**

## Why is this separate from core verification?

1. **Different use case**
   - Core: Verify existing text
   - Orchestrator: Generate + verify in real-time

2. **Complexity**
   - Core: 7 LLM calls, straightforward
   - Orchestrator: Multi-agent workflow, state management, skill delegation

3. **Maturity**
   - Core: Production-ready, tested (87.5% accuracy on HaluEval)
   - Orchestrator: Experimental, needs real API integration

4. **Dependencies**
   - Core: Just LLM provider
   - Orchestrator: LLM + web search + domain APIs + orchestration framework

## Example Usage

### NIL Player Report Generation

```python
from orchestrator.intelligent_orchestrator import IntelligentOrchestrator
from src.verification_pipeline import BedrockProvider

# Initialize
llm = BedrockProvider(region="us-east-1")
orch = IntelligentOrchestrator(llm_provider=llm)

# Generate verified NIL report
report = orch.generate_nil_report(
    player_name="Travis Hunter",
    position="WR/DB",
    school="Colorado",
    year=2024
)

# Check verification status
print(f"Overall confidence: {report.overall_confidence}")
for claim in report.verified_claims:
    print(f"- {claim.claim}: {claim.confidence}")
```

### CLI Tool

```bash
python orchestrator/examples/generate_nil_report.py \
  --player "Travis Hunter" \
  --school "Colorado" \
  --position "WR/DB" \
  --year 2024 \
  --output travis_hunter_report.json
```

## Evaluation Results

Tested on HaluEval QA hallucination detection:

**With LLM verification:**
- Accuracy: 87.5%
- Precision: 89.5%
- Recall: 85.0%
- F1 Score: 87.2%

**Baseline comparison:**
- Single-prompt baseline: 86.5% F1 (similar performance, simpler)
- Multi-step adds 0.7% F1 improvement (marginal)

**Key finding:** Current gains are primarily from model quality (Sonnet 4.5), not from multi-step reasoning.

See [BASELINE_COMPARISON.md](../archive/implementation_notes/BASELINE_COMPARISON.md) for full analysis.

## Research Questions

This orchestrator explores:

1. **Ghost Intelligence Error Correction**
   - Can we build reliable systems from unreliable AI?
   - What's the error correction "formula"?
   - Decomposition + Diverse Validators + External Grounding + Feedback?

2. **Verification During Generation vs After**
   - Is inline verification better than post-processing?
   - Does it prevent errors or just detect them differently?

3. **Multi-Model Verification**
   - Does using multiple LLMs reduce correlated errors?
   - What's the optimal ensemble strategy?

## Development Roadmap

### High Priority
- [ ] Integrate real NIL data APIs (ESPN, On3, Opendorse)
- [ ] Implement multi-model verification (Claude + GPT-4 + Gemini)
- [ ] Add feedback loops (log corrections, learn patterns)

### Medium Priority
- [ ] Complete CompletenessCheckSkill with LLM
- [ ] Add state management for retries/rollbacks
- [ ] Implement energy domain
- [ ] Implement medical domain

### Research
- [ ] Measure verification-during-generation vs post-processing
- [ ] Test diverse validator hypothesis (uncorrelated errors)
- [ ] Build confidence calibration from actual performance

## Documentation

- **Implementation notes:** See `../archive/implementation_notes/`
- **Evaluation results:** See `../archive/implementation_notes/HALUEVAL_RESULTS.md`
- **Baseline comparison:** See `../archive/implementation_notes/BASELINE_COMPARISON.md`

## Contributing

This is experimental research code. Contributions welcome but expect:
- Placeholder implementations
- Evolving APIs
- Research-focused, not production-optimized

For production verification needs, use the **core pipeline** instead.

---

## Quick Links

- 🏠 [Main README](../README.md) - Repository overview
- 🚀 [QUICKSTART](../QUICKSTART.md) - Core verification pipeline
- 📚 [Examples](../examples/) - Usage examples
- 📊 [Evaluation](../evaluation/) - Benchmarks and testing
- 📁 [Archive](../archive/) - Implementation history

---

**Status:** Experimental | **Maintainer:** Research team | **Last updated:** 2025-10-26
