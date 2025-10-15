# Interactive Verification Testing Guide

This guide explains how to use the interactive verification test system to walk through claim extraction and world model building step by step.

## Overview

The interactive test system provides a detailed walkthrough of the verification pipeline with focus on:
- **Claim Extraction**: Shows all extracted claims with their formal structures
- **World Model Building**: Step-by-step construction of the world state from propositions
- **Consistency Checking**: Real-time detection of contradictions and constraint violations
- **Verification Results**: Detailed breakdown of verification outcomes

## Quick Start

### 1. Basic Test (Simple Example)

```bash
# Activate virtual environment
source venv/bin/activate

# Run basic interactive test
python test_interactive_verification.py

# Or run without pauses (automated mode)
python test_interactive_verification.py --auto
```

### 2. Athlete Report Test (FLX Report)

```bash
# Show expected findings
python examples/athlete_report_test.py --expected

# Run with Mock provider (no actual LLM calls)
python examples/athlete_report_test.py --provider=mock

# Run with Anthropic Claude (requires API key)
export ANTHROPIC_API_KEY='your-key-here'
python examples/athlete_report_test.py --provider=anthropic

# Run with OpenAI GPT (requires API key)
export OPENAI_API_KEY='your-key-here'
python examples/athlete_report_test.py --provider=openai

# Run in automated mode (no pauses)
python examples/athlete_report_test.py --provider=anthropic --auto
```

## What to Expect

### Step 1: Input Report
Shows the original report text (either from JSON or direct text input).

**Example Output:**
```
================================================================================
STEP 1: INPUT REPORT
================================================================================

Athlete: Helaman Casuga
School: Corner Canyon (Utah)
Position: QB - Dual Threat
Physical: 6'0", 190 lbs
Performance: 4.72 sec 40-yard, 31" vertical
...
```

### Step 2: Claim Extraction
Shows all extracted claims with their metadata and formal structures.

**Example Output:**
```
================================================================================
STEP 2: CLAIM EXTRACTION
================================================================================
Found 23 claims (15 formalizable, 8 interpretive)

  CLAIM 1: "Helaman is 6'0\" tall"
    Type: QUANTITATIVE
    Source: basic_info
    Formalizable: YES
    Propositions:
      • {Helaman.height_inches = 72}

  CLAIM 2: "Weight is 190 lbs"
    Type: QUANTITATIVE
    Source: basic_info
    Formalizable: YES
    Propositions:
      • {Helaman.weight_lbs = 190}
...
```

### Step 3: Claim Classification
Separates claims into formalizable (→ world state) vs interpretive (→ LLM verification).

**Example Output:**
```
================================================================================
STEP 3: CLAIM CLASSIFICATION
================================================================================
Formalizable claims: 15
Interpretive claims: 8

--------------------------------------------------------------------------------
Formalizable Claims (→ World State Verification)
--------------------------------------------------------------------------------
  • Helaman is 6'0" tall
  • Weight is 190 lbs
  • 40-yard dash: 4.72 seconds
  • Protein: 165-190g per day
  ...

--------------------------------------------------------------------------------
Interpretive Claims (→ LLM Verification)
--------------------------------------------------------------------------------
  • Silent Assassin profile
  • Elite spatial awareness
  • Among rare <2% of high school players
  ...
```

### Step 4: World State Construction
Shows step-by-step building of the world model with propositions and constraints.

**Example Output:**
```
================================================================================
STEP 4: WORLD STATE CONSTRUCTION
================================================================================

--------------------------------------------------------------------------------
Building World State from Formalizable Claims
--------------------------------------------------------------------------------
Processing 15 formalizable claims...

  Processing: Helaman is 6'0" tall
    ✓ Added: Helaman.height_inches = 72

  Processing: Weight is 190 lbs
    ✓ Added: Helaman.weight_lbs = 190

  Processing: Protein distribution
    ✓ Added: Helaman.protein_per_day_min = 165
    ✓ Added: Helaman.protein_per_day_max = 190
    Constraint: protein_total / meals ≈ protein_per_meal
...

--------------------------------------------------------------------------------
Final World State
--------------------------------------------------------------------------------
================================================================================
WORLD STATE
================================================================================

Entities: 1
Propositions: 15
Constraints: 3
Issues: 0

Helaman:
  • height_inches: 72
  • weight_lbs: 190
  • forty_yard_seconds: 4.72
  • vertical_inches: 31
  • bench_press_lbs: 235
  • protein_per_day_min: 165
  • protein_per_day_max: 190
  ...

Constraints (3):
  1. protein_daily >= 165 AND protein_daily <= 190
  2. meals_per_day = protein_daily / protein_per_meal
  3. hydration_oz >= 80 AND hydration_oz <= 100

✓ No consistency issues detected
================================================================================
```

### Step 5: Consistency Analysis
Checks for contradictions and constraint violations.

**Example Output (if consistent):**
```
================================================================================
STEP 5: CONSISTENCY ANALYSIS
================================================================================
✓ World State: CONSISTENT
  Propositions: 15 added, 0 conflicts
  Constraints: 3 added, 0 violations
```

**Example Output (if inconsistent):**
```
================================================================================
STEP 5: CONSISTENCY ANALYSIS
================================================================================
❌ World State: INCONSISTENT (2 issues)
  1. Constraint violated: 50000000000 == 10 * 7000000000 (50B ≠ 70B)
  2. Contradictory values for Company.valuation: 50B vs 70B
```

### Step 6: Complete Verification
Shows full verification results with recommendations.

**Example Output:**
```
================================================================================
STEP 6: COMPLETE VERIFICATION
================================================================================
Running full verification pipeline...

--------------------------------------------------------------------------------
Verification Summary
--------------------------------------------------------------------------------
Total Claims: 23
✓ ├─ KEEP: 18 (high confidence)
⚠️  ├─ FLAG: 3 (moderate uncertainty)
❌ ├─ REVISE: 1 (issues found)
❌ └─ REMOVE: 1 (low confidence)

--------------------------------------------------------------------------------
Problematic Claims
--------------------------------------------------------------------------------

  ❌ Company X is valued at $50B
     Confidence: 0.00
     Recommendation: REVISE
     Issue: Constraint violated: 50B ≠ 10 * 7B
```

## Testing Interesting Scenarios

The athlete report has several interesting test scenarios:

### 1. Protein Math
**Claims:**
- Total protein: 165-190g/day
- Per meal: 30-45g protein
- Meals: 3+ meals per day

**Constraint Check:**
If 4 meals/day: 30-45g × 4 = 120-180g
This overlaps with 165-190g range → **CONSISTENT**

### 2. Comparable Builds
**Claims:**
- Helaman: 6'0" 190 lbs
- Marcel Reed: 6'1" 185 lbs (nearly identical)
- Jayden Daniels: slightly taller

**Question:** Are these comparisons mathematically reasonable?
- Height difference: 1 inch (1.4%)
- Weight difference: 5 lbs (2.6%)
Both within "nearly identical" range → **REASONABLE**

### 3. Performance Percentile
**Claim:** "Among rare <2% of high school players"

**Supporting Data:**
- 4.72 sec 40-yard
- 31" vertical
- 235 lbs bench

**Question:** Do these metrics actually put athlete in top 2%?
→ Requires external data/fact-checking

## Command Reference

### test_interactive_verification.py
```bash
python test_interactive_verification.py [--auto]

Options:
  --auto    Run without pausing between steps
```

### examples/athlete_report_test.py
```bash
python examples/athlete_report_test.py [OPTIONS]

Options:
  --report PATH      Path to athlete report JSON
                     (default: /Users/drw/cerbrec/code-conversion/resources/flx-resources/flx-report-payload.json)
  --provider TYPE    LLM provider: anthropic, openai, or mock
                     (default: mock)
  --auto            Run without pausing between steps
  --expected        Show expected findings and exit
```

## Troubleshooting

### No Claims Extracted
**Problem:** MockLLMProvider returns empty JSON
**Solution:** Use real LLM provider (Anthropic or OpenAI) with API key

### Import Errors
**Problem:** Module not found errors
**Solution:** Make sure virtual environment is activated:
```bash
source venv/bin/activate
```

### File Not Found
**Problem:** Report JSON file not found
**Solution:** Check the path and update with --report flag:
```bash
python examples/athlete_report_test.py --report=/path/to/your/report.json
```

## Next Steps

1. **Test with Real LLM**: Set API key and run with real provider
2. **Custom Reports**: Create your own JSON reports and test them
3. **Extend Visualization**: Modify display functions for your needs
4. **Add Constraints**: Enhance formal structure extraction for domain-specific constraints

## Example: Complete Walkthrough

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Check what we expect to find
python examples/athlete_report_test.py --expected

# 3. Run interactive test with Claude
export ANTHROPIC_API_KEY='your-key'
python examples/athlete_report_test.py --provider=anthropic

# 4. Walk through each step, pressing Enter to continue
# 5. Review the world state construction in detail
# 6. Check verification_report.json for full results
```

## Advanced Usage

### Custom Report Format
Create your own converter by extending `AthleteReportConverter`:

```python
class CustomReportConverter(AthleteReportConverter):
    @staticmethod
    def convert(report_data: Dict) -> str:
        # Your custom logic here
        return converted_text
```

### Custom Visualization
Modify display functions in `InteractiveDisplay`:

```python
display = InteractiveDisplay(auto_continue=False)
display.header("My Custom Section")
display.display_claims(claims)
```

### Programmatic Access
Use the pipeline directly for automation:

```python
from test_interactive_verification import InteractiveVerificationTest
from verification_pipeline import AnthropicProvider

llm = AnthropicProvider(api_key="your-key")
test = InteractiveVerificationTest(llm, auto_continue=True)
test.run_from_json("report.json", "Analyze report")
```
