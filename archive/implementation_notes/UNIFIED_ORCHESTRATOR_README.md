# Unified Intelligent Orchestrator - NIL Domain

## Overview

The Unified Intelligent Orchestrator integrates the 7-step intelligent workflow with rationality verification to generate **pre-verified intelligence reports**. Unlike traditional approaches where verification happens after generation, this system verifies claims **during generation** to catch errors before they compound.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│         INTELLIGENT REPORT ORCHESTRATOR                      │
├─────────────────────────────────────────────────────────────┤
│ Step 1: Goal Template + Domain Detection                    │
│ Step 2: Resolution Strategy + Verification Strategy         │
│ Step 3: Information Collection + FACT CHECKING              │
│ Step 4: Data Processing + WORLD STATE VERIFICATION          │
│ Step 5: Rational Connection + CONSISTENCY CHECKS            │
│ Step 6: AI Prediction + ADVERSARIAL REVIEW                  │
│ Step 7: Final Artifact + COMPLETENESS + SYNTHESIS           │
├─────────────────────────────────────────────────────────────┤
│         OUTPUT: Pre-Verified Intelligence Report             │
│         • Player analysis (verified stats)                   │
│         • NIL valuation (math-checked)                       │
│         • Predictions (adversarially reviewed)               │
│         • Confidence scores                                  │
└─────────────────────────────────────────────────────────────┘
```

## Key Innovation

**Verification-During-Generation** - Rationality checks are skills that agents use during report generation:
- Step 3 uses **FactCheckingSkill** to verify collected data
- Step 4 uses **WorldStateVerificationSkill** for mathematical consistency
- Step 6 uses **AdversarialReviewSkill** to challenge predictions
- Step 7 uses **SynthesisSkill** to combine all verification results

## Installation

### Requirements
- Python 3.10+
- Anthropic API key

### Setup

```bash
# Clone repo (if needed)
cd rationality-checks

# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your-key-here"
# Or add to .env file:
# ANTHROPIC_API_KEY=sk-ant-...
```

## File Structure

```
rationality-checks/
├── src/
│   ├── unified_orchestrator/
│   │   ├── __init__.py                         # Main exports
│   │   ├── intelligent_orchestrator.py         # 7-step orchestrator
│   │   ├── verification_skills.py              # Rationality checks as skills
│   │   └── domain_skills/
│   │       ├── base_domain.py                  # Abstract domain classes
│   │       └── nil_domain.py                   # NIL-specific skills
│   ├── verification_pipeline.py                # Existing (library)
│   ├── world_state_verification.py             # Existing (library)
│   └── integrated_verification.py              # Existing (library)
│
├── seven_steps/                                 # Original 7-step boilerplate
│   ├── main_orchestrator.py
│   ├── anthropic_skill_delegation_patterns.py
│   └── config.py
│
├── generate_nil_report.py                       # NIL report CLI
├── test_unified_integration.py                  # Integration tests
└── verify_document.py                           # Original document verifier
```

## Usage

### 1. Generate NIL Player Valuation Report

```bash
# Basic usage
python generate_nil_report.py \
  --player "Travis Hunter" \
  --school "Colorado" \
  --position "WR/DB"

# With output file
python generate_nil_report.py \
  --player "Shedeur Sanders" \
  --school "Colorado" \
  --position "QB" \
  --output sanders_report.json

# Custom query
python generate_nil_report.py \
  --query "Analyze top QB prospects for NIL value in 2024"

# Disable verification (faster, less accurate)
python generate_nil_report.py \
  --player "Player Name" \
  --school "School" \
  --no-verification
```

### 2. Programmatic Usage

```python
from src.unified_orchestrator import IntelligentOrchestrator

# Initialize orchestrator for NIL domain
orchestrator = IntelligentOrchestrator(
    domain="nil",
    model="claude-sonnet-4-5-20250929",
    enable_verification=True
)

# Generate report
report = orchestrator.generate_report(
    query="Evaluate Travis Hunter's NIL market value",
    context="Player: Travis Hunter, School: Colorado, Position: WR/DB, Season: 2024"
)

# Access results
print(f"Recommendation: {report.final_recommendation}")
print(f"Overall Confidence: {report.overall_confidence:.2f}")
print(f"Verifications: {report.verification_summary['total_verifications']}")
print(f"Issues Found: {report.verification_summary['issues_found']}")

# Check verification details
for verified_claim in report.verified_claims:
    if not verified_claim.passed:
        print(f"Failed: {verified_claim.claim}")
        print(f"  Issues: {verified_claim.issues}")
```

### 3. Run Integration Tests

```bash
# Run all integration tests
python test_unified_integration.py

# Expected output:
# ✓ All imports work
# ✓ Skills initialize correctly
# ✓ Tool definitions valid
# ✓ Skills execute properly
# ✓ World state verification works
```

## Components

### Verification Skills

**WorldStateVerificationSkill** - Mathematical consistency
- Builds formal world state from propositions
- Checks constraints and equations
- Confidence: 1.0 (passed) or 0.0 (failed) - mathematical proof

**FactCheckingSkill** - External fact verification
- Web search for supporting evidence
- Cross-checks claims against sources
- Confidence: Based on evidence strength

**EmpiricalTestingSkill** - Logical consistency
- Tests claims through scenarios
- Checks for contradictions
- Confidence: Based on consistency across tests

**AdversarialReviewSkill** - Challenge assumptions
- Identifies weak points
- Finds edge cases
- Proposes alternative interpretations

**CompletenessCheckSkill** - Identify gaps
- Missing context or caveats
- Unsupported assertions
- Incomplete analysis

**SynthesisSkill** - Combine results
- Aggregates all verification results
- Calculates overall confidence
- Makes final recommendation (keep/revise/remove/flag)

### NIL Domain Skills

**NILPlayerStatsSkill** - Fetch player statistics
- Sources: NCAA, ESPN, Sports-Reference
- Validates: Performance metrics
- Status: Placeholder (needs API integration)

**NILMarketDataSkill** - Fetch NIL valuations
- Sources: On3, Opendorse, 247Sports
- Provides: Market value, comparable players
- Status: Placeholder (needs API integration)

**NILPerformanceVerificationSkill** - Verify performance claims
- Cross-checks stats against official sources
- Detects fabricated/inflated statistics
- Status: Placeholder (needs NCAA API)

**NILValuationCheckSkill** - Validate valuations
- Mathematical consistency checks
- Comparable player analysis
- Status: Fully implemented

**NILTeamContextSkill** - Team information
- Rankings, conference, TV exposure
- Visibility multipliers for valuations
- Status: Placeholder (needs team data APIs)

## Verification Flow

### Step-by-Step Example: NIL Player Report

**Input:**
```
Query: "Evaluate Travis Hunter's NIL market value"
Context: "Colorado WR/DB, Heisman finalist 2024"
```

**Step 1: Goal Template**
- Objective: Determine player's NIL market value
- Domain: NIL
- Success criteria: Verified stats, math-checked valuation, evidence-based prediction

**Step 2: Strategy**
- Approach: Stats → Market comparison → Valuation formula → Prediction
- Verification checkpoints: Fact-check stats, validate calculations, challenge predictions

**Step 3: Information Collection + Fact Checking**
- Collects: Player stats, social metrics, market data
- **FactCheckingSkill**: Verifies stats against official sources
- Result: Confidence score for collected data

**Step 4: Data Processing + World State Verification**
- Processes: Normalize stats, calculate metrics
- **WorldStateVerificationSkill**: Math-checks valuation formula
  - Example: If "value = $500K" and "performance score = 5000" and "value = score × 100"
  - Checks: 500000 == 5000 × 100? ✓ Passes
- Result: Mathematically consistent data

**Step 5: Rational Connection + Consistency**
- Maps: Data → valuation factors
- **EmpiricalTestingSkill**: Checks logical consistency
- Result: Verified logical structure

**Step 6: AI Prediction + Adversarial Review**
- Predicts: Future NIL value growth
- **AdversarialReviewSkill**: Challenges assumptions
  - "What if player gets injured?"
  - "What if team performance declines?"
  - "What about market saturation?"
- Result: Prediction with risk factors

**Step 7: Final Artifact + Synthesis**
- **CompletenessCheckSkill**: Identifies missing caveats
- **SynthesisSkill**: Combines all verification results
- Result: Final report with overall confidence

**Output:**
```json
{
  "domain": "nil",
  "final_recommendation": "Travis Hunter's NIL value estimated at $500K-600K...",
  "overall_confidence": 0.85,
  "verification_summary": {
    "total_verifications": 8,
    "passed": 7,
    "failed": 1,
    "issues_found": 1
  },
  "verified_claims": [
    {
      "claim": "1200 receiving yards in 2024",
      "confidence": 0.95,
      "passed": true,
      "verification_methods": ["fact_check"]
    },
    {
      "claim": "NIL value $500K based on performance × social",
      "confidence": 1.0,
      "passed": true,
      "verification_methods": ["world_state"]
    }
  ]
}
```

## Current Limitations & Next Steps

### Implemented ✅
- [x] Unified orchestrator framework
- [x] Verification skills (6 skills)
- [x] NIL domain structure
- [x] Mathematical consistency verification
- [x] Valuation validation logic
- [x] CLI tool
- [x] Integration tests

### Needs Integration 🔧
- [ ] Real NCAA stats API (currently placeholder)
- [ ] NIL market data APIs (On3, Opendorse)
- [ ] Web search for fact-checking
- [ ] Social media metrics APIs

### Future Enhancements 🚀
- [ ] HaluEval dataset integration for evaluation
- [ ] Your real hallucination test cases
- [ ] Energy domain skills (manufacturing efficiency)
- [ ] Medical domain skills (healthcare information)
- [ ] Enhanced LLM integration for empirical testing
- [ ] Caching and performance optimization

## Testing & Evaluation Plan

### Phase 1: Integration Testing (Complete ✅)
```bash
python test_unified_integration.py
# All tests pass: 5/5
```

### Phase 2: Real Data Testing (Next)
1. **Add your real hallucination example**
   - Create test case in `tests/test_nil_hallucination.py`
   - Run through orchestrator
   - Verify it catches the hallucination

2. **Download HaluEval dataset**
   ```bash
   # Download prediction subset
   wget https://github.com/RUCAIBox/HaluEval/raw/main/data/qa_data.json
   mv qa_data.json evaluation/datasets/hallucination/
   ```

3. **Run baseline evaluation**
   ```python
   # Test on 20 HaluEval predictions
   # Measure: precision, recall, F1 for hallucination detection
   ```

### Phase 3: API Integration
1. Integrate NCAA stats API
2. Integrate On3 NIL valuations
3. Enable web search for fact-checking
4. Re-run evaluation with real data

## FAQ

**Q: Does this replace verify_document.py?**
A: No, they serve different purposes:
- `verify_document.py`: Verify existing documents (post-processing)
- `generate_nil_report.py`: Generate new reports with inline verification

**Q: Can I use this without the NIL domain?**
A: Yes, set `domain="generic"` for general-purpose reports.

**Q: How do I add a new domain (Energy, Medical)?**
A: Create `energy_domain.py` or `medical_domain.py` in `domain_skills/` following the NIL pattern.

**Q: Why are some skills showing placeholders?**
A: Data fetching skills need real APIs. Mathematical verification works fully without APIs.

**Q: How accurate is the verification?**
A:
- Mathematical verification: 100% accurate (formal proof)
- Fact-checking: Needs real data sources (currently placeholder)
- Logical consistency: Depends on LLM quality

**Q: Can I run this without API calls?**
A: Partially - mathematical verification works offline. Fact-checking and empirical testing need LLM/web access.

## Troubleshooting

**Import Error:**
```bash
# Make sure you're in the repo root
cd /Users/drw/cerbrec/rationality-checks
python generate_nil_report.py
```

**API Key Error:**
```bash
# Set environment variable
export ANTHROPIC_API_KEY="sk-ant-..."

# Or add to .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

**Test Failures:**
```bash
# Run integration tests to diagnose
python test_unified_integration.py

# Check specific import
python -c "from src.unified_orchestrator import IntelligentOrchestrator; print('OK')"
```

## Contributing

Priority areas:
1. **Add real hallucination test cases** - Your examples
2. **Integrate data APIs** - NCAA, On3, web search
3. **Add Energy domain** - Manufacturing efficiency analysis
4. **Add Medical domain** - Healthcare information verification
5. **Evaluation** - Run on HaluEval, measure performance

## Contact & Support

- Issues: Create issue in repo
- Questions: See main README.md for contact info
- Documentation: This file + code docstrings

---

**Status:** Phase 1 Complete (Foundation) ✅

**Next:** Add real test cases + integrate APIs + evaluate