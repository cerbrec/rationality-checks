 Unified Intelligent Orchestrator - NIL Domain Implementation Plan

 Architecture Overview

 Build a unified 7-step orchestrator that integrates rationality verification as skills/tools used during generation, not post-processing. Start with NIL (college player valuation) domain.

 Phase 1: Foundation - Unified Orchestrator Framework (Days 1-3)

 1.1 Create Directory Structure

 rationality-checks/
 ├── src/
 │   ├── unified_orchestrator/
 │   │   ├── __init__.py
 │   │   ├── intelligent_orchestrator.py    # Main unified orchestrator
 │   │   ├── verification_skills.py         # Rationality checks as skills
 │   │   ├── domain_skills/
 │   │   │   ├── __init__.py
 │   │   │   ├── base_domain.py            # Abstract domain skill classes
 │   │   │   └── nil_domain.py             # NIL-specific skills
 │   │   └── config.py                     # Unified config
 │   ├── [existing files remain unchanged]

 1.2 Convert Rationality Checks to Skills

 Create verification_skills.py with skill classes that wrap existing verification pipeline:

 Skills to Create:
 - WorldStateVerificationSkill - Wraps world_state_verification.py for mathematical consistency
 - EmpiricalTestingSkill - Logical consistency checks
 - FactCheckingSkill - Web search + external verification
 - AdversarialReviewSkill - Challenge assumptions and claims
 - CompletenessCheckSkill - Identify missing context/caveats
 - SynthesisSkill - Combine verification results

 Integration Pattern:
 class WorldStateVerificationSkill(Skill):
     """Skill that performs formal mathematical verification"""

     def get_tool_definition(self):
         return {
             "name": "verify_mathematical_consistency",
             "description": "Check mathematical consistency of claims",
             "input_schema": {...}
         }

     def execute(self, claims, **kwargs):
         # Call existing world_state_verification code
         verifier = WorldStateVerifier()
         return verifier.verify(claims)

 1.3 Create Intelligent Orchestrator

 Build intelligent_orchestrator.py that:
 - Inherits 7-step workflow structure from seven_steps/main_orchestrator.py
 - Equips each agent with appropriate verification skills
 - Routes to domain-specific skills based on context
 - Tracks verification results alongside generated content

 Modified Steps:
 1. Goal Template - Standard (classify domain: NIL/Energy/Medical)
 2. Resolution Strategy - Standard + select domain skills
 3. Information Collection - Uses FactCheckingSkill + domain data skills
 4. Data Processing - Uses WorldStateVerificationSkill
 5. Rational Connection - Uses EmpiricalTestingSkill
 6. AI Prediction - Uses AdversarialReviewSkill
 7. Final Artifact - Uses CompletenessCheckSkill + SynthesisSkill

 Phase 2: NIL Domain Implementation (Days 4-6)

 2.1 Create NIL Domain Skills (nil_domain.py)

 NILPlayerStatsSkill:
 - Fetch player statistics from sports data sources
 - Tools: NCAA stats, team websites, sports reference sites
 - Validates: Performance metrics (yards, TDs, tackles, etc.)

 NILMarketDataSkill:
 - Fetch NIL deal data and market valuations
 - Tools: On3 NIL Valuations, Opendorse, 247Sports
 - Validates: Market value claims against similar players

 NILPerformanceVerificationSkill:
 - Cross-check performance claims against official stats
 - Math validation: Stats calculations (yards/game, completion %, etc.)
 - Consistency: Season totals vs game-by-game

 NILValuationCheckSkill:
 - Validate valuation logic (performance + social + market)
 - Check math: Valuation formula consistency
 - Compare: Similar players with similar stats/following

 NILTeamContextSkill:
 - Fetch team information (rankings, conference, TV exposure)
 - Validate: Team performance claims
 - Context: School prestige, program visibility

 2.2 NIL-Specific Configuration

 Add to config.py:
 NIL_DOMAIN_CONFIG = {
     "data_sources": {
         "stats": ["NCAA", "ESPN", "Sports-Reference"],
         "nil_values": ["On3", "Opendorse"],
         "social": ["Twitter/X API", "Instagram API"]
     },
     "verification_thresholds": {
         "stats_confidence": 0.95,  # High - verifiable
         "valuation_confidence": 0.70,  # Medium - estimated
         "prediction_confidence": 0.60   # Lower - future-looking
     },
     "required_verifications": [
         "player_stats",
         "market_comparison",
         "calculation_consistency"
     ]
 }

 2.3 NIL Report Template

 Define structured output format:
 NILPlayerReport = {
     "player": {
         "name": str,
         "position": str,
         "school": str,
         "year": str
     },
     "performance": {
         "stats": {...},  # Verified ✓
         "verification_confidence": float
     },
     "valuation": {
         "estimated_value": float,  # Math-checked ✓
         "value_range": (float, float),
         "calculation_basis": str,
         "comparable_players": [...],  # Fact-checked ✓
         "verification_confidence": float
     },
     "prediction": {
         "future_value": float,  # Adversarially reviewed ✓
         "risk_factors": [...],
         "assumptions": [...],  # Completeness checked ✓
         "verification_confidence": float
     },
     "verification_summary": {
         "claims_verified": int,
         "issues_found": [...],
         "overall_confidence": float
     }
 }

 Phase 3: Evaluation Setup (Days 7-8)

 3.1 Download Hallucination Datasets

 - HaluEval QA subset (focus on prediction tasks)
 - TruthfulQA (for factual claims)
 - Store in evaluation/datasets/hallucination/

 3.2 Create NIL Test Cases

 File: tests/test_nil_domain.py

 Test Cases:
 1. Your Real Hallucination Example - Most important!
 2. Known Player Example (Travis Hunter 2024):
   - Stats: Verified via NCAA
   - NIL Value: Known estimates
   - Test if system catches fabricated stats
 3. Synthetic Test Cases:
   - Correct stats + correct valuation
   - Correct stats + inflated valuation (should catch math error)
   - Fabricated stats + any valuation (should catch via fact-check)
   - Logical contradictions (e.g., "QB with 1200 receiving yards")

 3.3 Evaluation Metrics

 Measure:
 - Hallucination Detection Rate: % of fabricated claims caught
 - False Positive Rate: % of correct claims incorrectly flagged
 - Domain Accuracy: Correct stat verification, valuation checks
 - Step-wise Analysis: Which steps catch which types of errors

 Phase 4: CLI & Integration (Days 9-10)

 4.1 Create NIL CLI Tool

 File: generate_nil_report.py

 python generate_nil_report.py \
   --player "Travis Hunter" \
   --school "Colorado" \
   --position "WR/DB" \
   --year "2024" \
   --output travis_hunter_report.json \
   --verbose

 # Generates report with:
 # - Verified stats from NCAA/ESPN
 # - Market valuation with confidence scores
 # - Comparable players (verified)
 # - Future value prediction (adversarially reviewed)
 # - All claims tagged with verification status

 4.2 Update verify_document.py

 Add domain detection:
 # If document contains NIL keywords → use NIL domain config
 # Still works for generic documents (backward compatible)

 Phase 5: Testing & Iteration (Days 11-14)

 5.1 Test with Your Real Hallucination

 1. Run through unified orchestrator
 2. Check if verification catches the hallucination
 3. If not → document why → enhance specific skill
 4. Iterate until caught

 5.2 Benchmark Evaluation

 1. Run 20 NIL reports through system
 2. Test against HaluEval prediction subset
 3. Measure detection rates
 4. Compare: Unified vs separate verification approach

 5.3 Documentation

 Create docs/NIL_DOMAIN_GUIDE.md:
 - How to generate NIL reports
 - Data sources used
 - Verification methodology
 - Confidence score interpretation
 - Example outputs

 Deliverables

 1. ✅ Unified Orchestrator Framework
   - verification_skills.py (6 skill classes)
   - intelligent_orchestrator.py (integrated 7-step + verification)
 2. ✅ NIL Domain Implementation
   - nil_domain.py (5 NIL-specific skills)
   - NIL domain configuration
   - NIL report template
 3. ✅ Evaluation Suite
   - HaluEval dataset integration
   - NIL test cases (including your real example)
   - Evaluation metrics script
 4. ✅ CLI Tools
   - generate_nil_report.py (NIL-specific)
   - Enhanced verify_document.py (domain-aware)
 5. ✅ Documentation
   - Architecture overview
   - NIL domain guide
   - API reference
   - Evaluation results

 Success Criteria

 - Primary: Catches your real hallucination example (100%)
 - Stats Verification: >95% accuracy on NCAA stats claims
 - Valuation Checks: Catches 80%+ of math errors in valuations
 - Prediction Review: Flags unrealistic predictions (>70%)
 - Overall: >75% hallucination detection on HaluEval predictions
 - Usability: Single command generates verified NIL report

 First Steps to Execute

 1. Create src/unified_orchestrator/ directory structure
 2. Copy seven_steps/main_orchestrator.py as base template
 3. Create verification_skills.py with WorldStateVerificationSkill
 4. Test skill wrapping of existing world_state_verification.py
 5. Add your real hallucination example to tests/
 6. Run baseline test: Does current system catch it?

