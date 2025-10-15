"""
Interactive Verification Test System

Provides step-by-step walkthrough of the verification pipeline with
detailed display of claim extraction and world model building.
"""

import json
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from integrated_verification import IntegratedVerificationPipeline, EnhancedClaim
from verification_pipeline import LLMProvider, ClaimType
from world_state_verification import WorldState, Proposition, format_value


# ============================================================================
# JSON TO TEXT CONVERTER
# ============================================================================

class AthleteReportConverter:
    """Converts structured athlete report JSON to narrative text"""

    @staticmethod
    def convert(report_data: Dict) -> str:
        """Convert JSON report to readable narrative text"""
        lines = []
        content = report_data.get("content", {})

        # Header
        lines.append("=" * 80)
        lines.append("ATHLETE PERFORMANCE REPORT")
        lines.append("=" * 80)

        # Basic Info
        basic_info = content.get("basic_info", [])
        if basic_info:
            lines.append("\nBASIC INFORMATION:")
            for item in basic_info:
                lines.append(f"{item['label']}: {item['value']}")

        # Performance Metrics
        metrics = content.get("performance_metrics", [])
        if metrics:
            lines.append("\nPERFORMANCE METRICS:")
            for item in metrics:
                lines.append(f"{item['label']}: {item['value']}")

        # Grit Profile
        grit = content.get("grit_profile", {})
        if grit:
            lines.append(f"\nGRIT PROFILE:")
            lines.append(f"Type: {grit.get('profile_type')}")
            lines.append(f"Description: {grit.get('description')}")

        # Who You Are
        who_you_are = content.get("who_you_are", {})
        if who_you_are:
            lines.append("\nANALYSIS:")
            for para in who_you_are.get("paragraphs", []):
                lines.append(f"\n{para}")

        # Frame Projection
        frame = content.get("frame_projection", {})
        if frame:
            lines.append("\nFRAME PROJECTION:")
            for para in frame.get("paragraphs", []):
                lines.append(f"\n{para}")

        # Comparable Builds
        comparables = content.get("comparable_builds", [])
        if comparables:
            lines.append("\nCOMPARABLE BUILDS:")
            for comp in comparables:
                lines.append(f"\n• {comp['name']}: {comp['description']}")

        # Nutrition
        nutrition = content.get("nutrition", {})
        if nutrition:
            lines.append("\nNUTRITION PLAN:")
            macros = nutrition.get("macronutrients_block", {})
            if macros:
                protein = macros.get("protein", {})
                carbs = macros.get("carbohydrates", {})
                fats = macros.get("fats", {})

                if protein:
                    lines.append(f"Protein: {protein.get('value')}")
                if carbs:
                    lines.append(f"Carbohydrates: {carbs.get('value')}")
                if fats:
                    lines.append(f"Fats: {fats.get('value')}")

                # Add detailed nutrition info
                if protein:
                    learn_more = protein.get("learn_more", {})
                    content_block = learn_more.get("content", {})
                    instructions = content_block.get("instructions_block", {})
                    if instructions:
                        instructions_list = instructions.get("instructions", [])
                        if instructions_list:
                            lines.append("\nProtein Distribution:")
                            for inst in instructions_list:
                                lines.append(f"  • {inst['label']}: {inst['value']}")

        return "\n".join(lines)


# ============================================================================
# INTERACTIVE DISPLAY FUNCTIONS
# ============================================================================

class InteractiveDisplay:
    """Handles step-by-step display with pauses"""

    def __init__(self, auto_continue: bool = False):
        self.auto_continue = auto_continue
        self.step_number = 0

    def header(self, title: str):
        """Display a section header"""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    def section(self, title: str):
        """Display a sub-section"""
        print("\n" + "-" * 80)
        print(title)
        print("-" * 80)

    def step(self, description: str):
        """Display a step"""
        self.step_number += 1
        print(f"\n[STEP {self.step_number}] {description}")

    def info(self, message: str, indent: int = 0):
        """Display information"""
        prefix = "  " * indent
        print(f"{prefix}{message}")

    def success(self, message: str):
        """Display success message"""
        print(f"✓ {message}")

    def warning(self, message: str):
        """Display warning message"""
        print(f"⚠️  {message}")

    def error(self, message: str):
        """Display error message"""
        print(f"❌ {message}")

    def pause(self, message: str = "Press Enter to continue..."):
        """Pause for user input"""
        if not self.auto_continue:
            input(f"\n{message}")

    def display_claims(self, claims: List[EnhancedClaim]):
        """Display extracted claims with formatting"""
        formal_count = sum(1 for c in claims if c.is_formalizable)
        interpretive_count = len(claims) - formal_count

        self.info(f"Found {len(claims)} claims ({formal_count} formalizable, {interpretive_count} interpretive)")

        for i, claim in enumerate(claims, 1):
            print(f"\n  CLAIM {i}: \"{claim.text}\"")
            print(f"    Type: {claim.claim_type.value.upper()}")
            print(f"    Source: {claim.source_section}")
            print(f"    Formalizable: {'YES' if claim.is_formalizable else 'NO'}")

            if claim.is_formalizable and claim.formal_structure:
                self.display_formal_structure(claim.formal_structure, indent=2)

    def display_formal_structure(self, structure: Dict, indent: int = 0):
        """Display formal structure of a claim"""
        prefix = "  " * indent

        propositions = structure.get("propositions", [])
        if propositions:
            print(f"{prefix}Propositions:")
            for prop in propositions:
                value_str = format_value(prop.get("value"))
                print(f"{prefix}  • {{{prop.get('subject')}.{prop.get('predicate')} = {value_str}}}")

        constraints = structure.get("constraints", [])
        if constraints:
            print(f"{prefix}Constraints:")
            for const in constraints:
                print(f"{prefix}  • {const.get('formula')}")

        implications = structure.get("implications", [])
        if implications:
            print(f"{prefix}Implications:")
            for imp in implications:
                print(f"{prefix}  • {imp}")

    def display_world_state_building(self, claims: List[EnhancedClaim], world: WorldState):
        """Display the world state building process step by step"""
        self.section("Building World State from Formalizable Claims")

        formal_claims = [c for c in claims if c.is_formalizable]
        self.info(f"Processing {len(formal_claims)} formalizable claims...")

        for claim in formal_claims:
            if not claim.formal_structure:
                continue

            print(f"\n  Processing: {claim.text}")

            # Show propositions being added
            props = claim.formal_structure.get("propositions", [])
            for prop_data in props:
                subject = prop_data.get("subject")
                predicate = prop_data.get("predicate")
                value = prop_data.get("value")
                value_str = format_value(value)

                # Check if it conflicts
                existing = world.get_proposition(subject, predicate)
                if existing and existing.value != value:
                    self.error(f"    Conflict! {subject}.{predicate}: {existing.value} vs {value}")
                else:
                    self.success(f"    Added: {subject}.{predicate} = {value_str}")

            # Show constraints being added
            constraints = claim.formal_structure.get("constraints", [])
            for const_data in constraints:
                formula = const_data.get("formula")
                self.info(f"    Constraint: {formula}", indent=1)


# ============================================================================
# MAIN INTERACTIVE TEST
# ============================================================================

class InteractiveVerificationTest:
    """Main interactive test controller"""

    def __init__(self, llm_provider: LLMProvider, auto_continue: bool = False):
        self.pipeline = IntegratedVerificationPipeline(llm_provider)
        self.display = InteractiveDisplay(auto_continue)

    def run_from_json(self, json_path: str, query: str = "Analyze athlete report"):
        """Run interactive test from JSON file"""
        # Load JSON
        with open(json_path, 'r') as f:
            report_data = json.load(f)

        # Convert to text
        converter = AthleteReportConverter()
        report_text = converter.convert(report_data)

        # Run interactive test
        self.run_from_text(report_text, query, show_input=True)

    def run_from_text(
        self,
        report_text: str,
        query: str,
        show_input: bool = True
    ):
        """Run interactive test from text"""

        # Step 1: Show Input
        if show_input:
            self.display.header("STEP 1: INPUT REPORT")
            print(report_text[:2000] + "..." if len(report_text) > 2000 else report_text)
            self.display.pause()

        # Step 2: Claim Extraction
        self.display.header("STEP 2: CLAIM EXTRACTION")
        self.display.info("Extracting claims and formal structures...")

        claims = self.pipeline._extract_claims_with_structure(report_text, query)

        self.display.display_claims(claims)
        self.display.pause()

        # Step 3: Separate Claims
        self.display.header("STEP 3: CLAIM CLASSIFICATION")

        formal_claims = [c for c in claims if c.is_formalizable]
        interpretive_claims = [c for c in claims if not c.is_formalizable]

        self.display.info(f"Formalizable claims: {len(formal_claims)}")
        self.display.info(f"Interpretive claims: {len(interpretive_claims)}")

        self.display.section("Formalizable Claims (→ World State Verification)")
        for claim in formal_claims:
            self.display.info(f"• {claim.text}", indent=1)

        self.display.section("Interpretive Claims (→ LLM Verification)")
        for claim in interpretive_claims:
            self.display.info(f"• {claim.text}", indent=1)

        self.display.pause()

        # Step 4: World State Building
        if formal_claims:
            self.display.header("STEP 4: WORLD STATE CONSTRUCTION")

            # Build world state
            world_results, world_state = self.pipeline._world_state_verify(formal_claims)

            self.display.display_world_state_building(formal_claims, world_state)

            # Show final world state
            self.display.section("Final World State")
            print(world_state.visualize(verbose=False))

            self.display.pause()

        # Step 5: Consistency Analysis
        self.display.header("STEP 5: CONSISTENCY ANALYSIS")

        if formal_claims:
            is_consistent, issues = world_state.is_consistent()

            if is_consistent:
                self.display.success(f"World State: CONSISTENT")
                self.display.info(f"Propositions: {len(world_state.propositions)} added, 0 conflicts")
                self.display.info(f"Constraints: {len(world_state.constraints)} added, 0 violations")
            else:
                self.display.error(f"World State: INCONSISTENT ({len(issues)} issues)")
                for i, issue in enumerate(issues, 1):
                    self.display.error(f"{i}. {issue.description}")
        else:
            self.display.info("No formalizable claims to verify")

        self.display.pause()

        # Step 6: Full Verification
        self.display.header("STEP 6: COMPLETE VERIFICATION")
        self.display.info("Running full verification pipeline...")

        report = self.pipeline.verify_analysis(report_text, query)

        # Display summary
        self.display.section("Verification Summary")

        keep_count = sum(1 for a in report.assessments if a.recommendation == "keep")
        flag_count = sum(1 for a in report.assessments if a.recommendation == "flag_uncertainty")
        revise_count = sum(1 for a in report.assessments if a.recommendation == "revise")
        remove_count = sum(1 for a in report.assessments if a.recommendation == "remove")

        self.display.info(f"Total Claims: {len(report.assessments)}")
        self.display.success(f"├─ KEEP: {keep_count} (high confidence)")
        self.display.warning(f"├─ FLAG: {flag_count} (moderate uncertainty)")
        self.display.error(f"├─ REVISE: {revise_count} (issues found)")
        self.display.error(f"└─ REMOVE: {remove_count} (low confidence)")

        # Show detailed results for problematic claims
        problematic = [a for a in report.assessments if a.recommendation in ["revise", "remove"]]
        if problematic:
            self.display.section("Problematic Claims")
            for assessment in problematic:
                print(f"\n  ❌ {assessment.claim.text}")
                print(f"     Confidence: {assessment.overall_confidence:.2f}")
                print(f"     Recommendation: {assessment.recommendation.upper()}")

                for result in assessment.verification_results:
                    if not result.passed and result.issues_found:
                        for issue in result.issues_found:
                            print(f"     Issue: {issue}")

        self.display.header("VERIFICATION COMPLETE")
        print(f"\nFull report saved to: verification_report.json")

        # Save report
        with open("verification_report.json", "w") as f:
            json.dump({
                "summary": report.summary,
                "total_claims": len(report.assessments),
                "recommendations": {
                    "keep": keep_count,
                    "flag": flag_count,
                    "revise": revise_count,
                    "remove": remove_count
                }
            }, f, indent=2)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage"""
    import argparse
    from verification_pipeline import MockLLMProvider

    parser = argparse.ArgumentParser(description="Interactive verification test")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-continue without pauses"
    )
    args = parser.parse_args()

    # For real testing, use:
    # from verification_pipeline import AnthropicProvider
    # llm = AnthropicProvider(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    llm = MockLLMProvider()

    test = InteractiveVerificationTest(llm, auto_continue=args.auto)

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                INTERACTIVE VERIFICATION TEST SYSTEM                       ║
║                                                                           ║
║  This system provides a step-by-step walkthrough of the verification     ║
║  pipeline, with special focus on:                                        ║
║  • Claim extraction with formal structure                                ║
║  • World model building from propositions                                ║
║  • Consistency checking and contradiction detection                      ║
║                                                                           ║
║  Usage:                                                                   ║
║    test.run_from_json(json_path, query)                                  ║
║    test.run_from_text(text, query)                                       ║
║                                                                           ║
║  Options:                                                                 ║
║    --auto    Run without pausing (for automated testing)                 ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    # Example with simple text
    example_text = """
    Athlete Profile: John Doe
    Height: 6'2", Weight: 195 lbs
    40-yard dash: 4.65 seconds
    Vertical jump: 36 inches

    Analysis: John is among the top 5% of athletes in his class. His combination
    of size and speed makes him an elite prospect. Based on his metrics, he projects
    as a Division I athlete with potential for professional development.

    Training Plan:
    - Protein intake: 180-200g per day
    - 4-5 meals per day at 35-45g protein each
    - Strength training 4x per week
    """

    test.run_from_text(example_text, "Analyze athlete profile")


if __name__ == "__main__":
    main()
