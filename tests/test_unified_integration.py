#!/usr/bin/env python3
"""
Test Unified Orchestrator Integration
======================================

Quick test to verify that all modules integrate correctly.
Tests imports, skill initialization, and basic functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from src.unified_orchestrator import IntelligentOrchestrator
        print("  ✓ IntelligentOrchestrator imported")
    except ImportError as e:
        print(f"  ✗ Failed to import IntelligentOrchestrator: {e}")
        return False

    try:
        from src.unified_orchestrator.verification_skills import (
            WorldStateVerificationSkill,
            FactCheckingSkill,
            EmpiricalTestingSkill,
            AdversarialReviewSkill,
            CompletenessCheckSkill,
            SynthesisSkill
        )
        print("  ✓ Verification skills imported")
    except ImportError as e:
        print(f"  ✗ Failed to import verification skills: {e}")
        return False

    try:
        from src.unified_orchestrator.domain_skills.nil_domain import (
            NILPlayerStatsSkill,
            NILMarketDataSkill,
            NILPerformanceVerificationSkill,
            NILValuationCheckSkill,
            NILTeamContextSkill,
            NIL_DOMAIN_CONFIG
        )
        print("  ✓ NIL domain skills imported")
    except ImportError as e:
        print(f"  ✗ Failed to import NIL domain skills: {e}")
        return False

    return True


def test_skill_initialization():
    """Test that skills can be initialized"""
    print("\nTesting skill initialization...")

    from src.unified_orchestrator.verification_skills import (
        WorldStateVerificationSkill,
        FactCheckingSkill,
        SynthesisSkill
    )
    from src.unified_orchestrator.domain_skills.nil_domain import (
        NILPlayerStatsSkill,
        NILValuationCheckSkill
    )

    try:
        # Test verification skills
        world_state_skill = WorldStateVerificationSkill()
        print("  ✓ WorldStateVerificationSkill initialized")

        fact_check_skill = FactCheckingSkill()
        print("  ✓ FactCheckingSkill initialized")

        synthesis_skill = SynthesisSkill()
        print("  ✓ SynthesisSkill initialized")

        # Test NIL domain skills
        player_stats_skill = NILPlayerStatsSkill()
        print("  ✓ NILPlayerStatsSkill initialized")

        valuation_skill = NILValuationCheckSkill()
        print("  ✓ NILValuationCheckSkill initialized")

        return True
    except Exception as e:
        print(f"  ✗ Failed to initialize skills: {e}")
        return False


def test_tool_definitions():
    """Test that skills return proper tool definitions"""
    print("\nTesting tool definitions...")

    from src.unified_orchestrator.verification_skills import WorldStateVerificationSkill
    from src.unified_orchestrator.domain_skills.nil_domain import NILPlayerStatsSkill

    try:
        world_state_skill = WorldStateVerificationSkill()
        tool_def = world_state_skill.get_tool_definition()

        assert "name" in tool_def
        assert "description" in tool_def
        assert "input_schema" in tool_def
        print(f"  ✓ WorldStateVerificationSkill tool definition: {tool_def['name']}")

        player_stats_skill = NILPlayerStatsSkill()
        tool_def = player_stats_skill.get_tool_definition()

        assert "name" in tool_def
        assert "description" in tool_def
        assert "input_schema" in tool_def
        print(f"  ✓ NILPlayerStatsSkill tool definition: {tool_def['name']}")

        return True
    except Exception as e:
        print(f"  ✗ Failed tool definition test: {e}")
        return False


def test_skill_execution():
    """Test basic skill execution (without API)"""
    print("\nTesting skill execution...")

    from src.unified_orchestrator.verification_skills import SynthesisSkill
    from src.unified_orchestrator.domain_skills.nil_domain import NILValuationCheckSkill

    try:
        # Test synthesis skill with mock data
        synthesis_skill = SynthesisSkill()
        verification_results = [
            {"passed": True, "confidence": 0.9, "issues": []},
            {"passed": True, "confidence": 0.8, "issues": []},
            {"passed": False, "confidence": 0.3, "issues": ["Math error"]}
        ]

        result = synthesis_skill.execute(verification_results=verification_results)
        assert result["status"] == "success"
        assert "overall_confidence" in result
        assert "recommendation" in result
        print(f"  ✓ SynthesisSkill execution: confidence={result['overall_confidence']:.2f}, rec={result['recommendation']}")

        # Test NIL valuation skill
        valuation_skill = NILValuationCheckSkill()
        data = {
            "estimated_value": 500000,
            "valuation_factors": {"performance": 0.8, "social": 0.7},
            "comparable_players": [
                {"nil_value": 480000},
                {"nil_value": 520000}
            ]
        }

        result = valuation_skill.execute(data=data)
        assert result["status"] == "success"
        assert "passed" in result
        print(f"  ✓ NILValuationCheckSkill execution: passed={result['passed']}, confidence={result['confidence']:.2f}")

        return True
    except Exception as e:
        print(f"  ✗ Failed skill execution test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_world_state_integration():
    """Test that WorldStateVerificationSkill properly wraps existing code"""
    print("\nTesting world state integration...")

    from src.unified_orchestrator.verification_skills import WorldStateVerificationSkill

    try:
        skill = WorldStateVerificationSkill()

        # Test with simple quantitative claims
        claims = [
            {
                "id": "claim1",
                "text": "Player value is $500K",
                "propositions": [
                    {"subject": "Player", "predicate": "value", "value": 500000}
                ],
                "constraints": []
            },
            {
                "id": "claim2",
                "text": "Player value equals performance * 100",
                "propositions": [
                    {"subject": "Player", "predicate": "performance", "value": 5000}
                ],
                "constraints": [
                    {"variables": ["Player.value", "Player.performance"], "formula": "Player.value == Player.performance * 100"}
                ]
            }
        ]

        result = skill.execute(claims=claims)
        assert result["status"] == "success"
        assert "passed" in result
        assert "confidence" in result
        print(f"  ✓ World state verification: passed={result['passed']}, confidence={result['confidence']}")

        return True
    except Exception as e:
        print(f"  ✗ Failed world state integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("UNIFIED ORCHESTRATOR INTEGRATION TESTS")
    print("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("Skill Initialization", test_skill_initialization),
        ("Tool Definitions", test_tool_definitions),
        ("Skill Execution", test_skill_execution),
        ("World State Integration", test_world_state_integration),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    if failed == 0:
        print("\n✅ All tests passed!")
        print("\nNext steps:")
        print("1. Set ANTHROPIC_API_KEY in .env file")
        print("2. Run: python generate_nil_report.py --query 'Test NIL report'")
        print("3. Check output for verification results")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        print("Review errors above and fix before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
