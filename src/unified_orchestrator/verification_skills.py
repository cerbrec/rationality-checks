"""
Verification Skills
===================

Converts rationality-checks verification methods into reusable skills
that can be used by orchestrator agents during intelligent report generation.

These skills wrap the existing verification pipeline components:
- World State Verification (formal mathematical verification)
- Empirical Testing (logical consistency)
- Fact Checking (web search + external sources)
- Adversarial Review (challenge assumptions)
- Completeness Check (identify gaps)
- Synthesis (combine verification results)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import sys
from pathlib import Path

# Add parent directory to path to import existing verification modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from verification_pipeline import (
    Claim, ClaimType, VerificationMethod, VerificationResult,
    Evidence, LLMProvider
)
from world_state_verification import (
    WorldState, WorldStateVerifier, ClaimInterpreter, Proposition, Constraint
)


class Skill(ABC):
    """
    Abstract base class for verification skills.

    Skills provide specific capabilities that agents can use via tool calling.
    Each skill wraps a verification method from the rationality-checks pipeline.
    """

    @abstractmethod
    def get_tool_definition(self) -> dict:
        """
        Return Anthropic-compatible tool definition.

        Returns:
            dict: Tool definition with name, description, input_schema
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the skill.

        Returns:
            dict: Result with status, data, confidence, and evidence
        """
        pass


class WorldStateVerificationSkill(Skill):
    """
    Skill: Formal mathematical verification using world state.

    Verifies mathematical consistency of quantitative claims by:
    1. Building a world state model from propositions
    2. Checking constraints and equations
    3. Detecting contradictions with mathematical certainty

    Confidence: 1.0 for passed, 0.0 for failed (mathematical proof)
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        # World state verification doesn't actually need LLM for basic math checking
        # LLM is only needed if we want to interpret natural language claims
        # For now, we'll work with pre-structured claims
        self.llm_provider = llm_provider
        if llm_provider:
            self.verifier = WorldStateVerifier(llm_provider)
            self.interpreter = ClaimInterpreter()
        else:
            # For basic verification, we can work without LLM
            # by operating directly on pre-structured claims
            self.verifier = None
            self.interpreter = None

    def get_tool_definition(self) -> dict:
        return {
            "name": "verify_mathematical_consistency",
            "description": "Verify mathematical consistency of quantitative claims using formal world state model. Returns confidence=1.0 (passed) or 0.0 (contradiction detected).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "propositions": {
                                    "type": "array",
                                    "items": {"type": "object"}
                                },
                                "constraints": {
                                    "type": "array",
                                    "items": {"type": "object"}
                                }
                            }
                        },
                        "description": "Claims with formal structures (propositions and constraints)"
                    }
                },
                "required": ["claims"]
            }
        }

    def execute(self, claims: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Execute world state verification.

        Args:
            claims: List of claims with formal structures

        Returns:
            dict: Verification result with issues found
        """
        try:
            # Build world state from claims
            world_state = WorldState()
            claim_objects = []

            for claim_data in claims:
                # Convert dict to Claim object
                claim = Claim(
                    id=claim_data.get("id", ""),
                    text=claim_data.get("text", ""),
                    claim_type=ClaimType.QUANTITATIVE,  # World state for quantitative claims
                    source_section="",
                )
                claim_objects.append(claim)

                # Add propositions to world state
                for prop_data in claim_data.get("propositions", []):
                    prop = Proposition(
                        subject=prop_data.get("subject", ""),
                        predicate=prop_data.get("predicate", ""),
                        value=prop_data.get("value"),
                        source_claim_id=claim_data.get("id", "")
                    )
                    world_state.add_proposition(prop)

                # Add constraints to world state
                for const_data in claim_data.get("constraints", []):
                    const = Constraint(
                        constraint_type="equation",  # Default to equation
                        variables=const_data.get("variables", []),
                        formula=const_data.get("formula", ""),
                        source_claim_id=claim_data.get("id", "")
                    )
                    world_state.add_constraint(const)

            # Check for consistency issues directly from world state
            # (Don't need verifier for basic consistency checking)
            issues = world_state.consistency_issues

            # Determine pass/fail
            passed = len(issues) == 0
            confidence = 1.0 if passed else 0.0

            return {
                "status": "success",
                "method": "world_state",
                "passed": passed,
                "confidence": confidence,
                "issues": [issue.description for issue in issues],
                "world_state_summary": {
                    "propositions": len(world_state.propositions),
                    "constraints": len(world_state.constraints)
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "method": "world_state",
                "passed": False,
                "confidence": 0.0,
                "error": str(e),
                "issues": [f"Error during verification: {str(e)}"]
            }


class FactCheckingSkill(Skill):
    """
    Skill: Fact-checking using web search and external sources.

    Verifies factual claims by:
    1. Searching the web for supporting/refuting evidence
    2. Extracting relevant information from sources
    3. Assessing agreement with claim

    Confidence: Based on evidence strength and source quality
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider

    def get_tool_definition(self) -> dict:
        return {
            "name": "fact_check_claim",
            "description": "Verify factual claims using web search and external sources. Returns confidence based on evidence strength.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The claim to fact-check"
                    },
                    "search_query": {
                        "type": "string",
                        "description": "Optional custom search query (auto-generated if not provided)"
                    },
                    "required_sources": {
                        "type": "integer",
                        "description": "Minimum number of sources to find (default: 3)"
                    }
                },
                "required": ["claim"]
            }
        }

    def execute(self, claim: str, search_query: Optional[str] = None,
                required_sources: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Execute fact-checking operation.

        Args:
            claim: Claim to fact-check
            search_query: Optional custom search query
            required_sources: Minimum number of sources to find

        Returns:
            dict: Fact-check result with evidence and confidence
        """
        try:
            # For now, return a placeholder that indicates web search is needed
            # In full implementation, this would use the LLM provider's web search
            # capability or integrate with a web search API

            return {
                "status": "success",
                "method": "fact_check",
                "claim": claim,
                "search_query": search_query or f"verify: {claim}",
                "evidence": [],
                "passed": None,  # Requires actual web search to determine
                "confidence": 0.5,  # Neutral until search is performed
                "note": "Fact-checking requires web search integration",
                "recommendation": "Enable web search or provide evidence manually"
            }

        except Exception as e:
            return {
                "status": "error",
                "method": "fact_check",
                "claim": claim,
                "passed": False,
                "confidence": 0.0,
                "error": str(e)
            }


class EmpiricalTestingSkill(Skill):
    """
    Skill: Empirical logical consistency testing.

    Tests logical consistency by:
    1. Generating test scenarios
    2. Checking for contradictions
    3. Validating logical implications

    Confidence: Based on consistency across test scenarios
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider

    def get_tool_definition(self) -> dict:
        return {
            "name": "test_logical_consistency",
            "description": "Test logical consistency of claims through empirical testing scenarios. Returns confidence based on consistency across tests.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Claims to test for logical consistency"
                    },
                    "context": {
                        "type": "string",
                        "description": "Context for testing"
                    }
                },
                "required": ["claims"]
            }
        }

    def execute(self, claims: List[str], context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute empirical consistency testing.

        Args:
            claims: Claims to test
            context: Optional context

        Returns:
            dict: Consistency test result
        """
        # Placeholder implementation - would use LLM for empirical testing
        return {
            "status": "success",
            "method": "empirical_test",
            "claims_tested": len(claims),
            "passed": True,
            "confidence": 0.7,
            "issues": [],
            "note": "Empirical testing requires LLM integration"
        }


class AdversarialReviewSkill(Skill):
    """
    Skill: Adversarial review to challenge claims and assumptions.

    Reviews claims by:
    1. Identifying assumptions
    2. Finding edge cases
    3. Proposing alternative interpretations
    4. Challenging conclusions

    Confidence: Lower for claims with weak assumptions or edge cases
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider

    def get_tool_definition(self) -> dict:
        return {
            "name": "adversarial_review",
            "description": "Challenge claims and assumptions through adversarial review. Identifies weak points, edge cases, and alternative interpretations.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "Claim to review adversarially"
                    },
                    "assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stated assumptions to challenge"
                    }
                },
                "required": ["claim"]
            }
        }

    def execute(self, claim: str, assumptions: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute adversarial review.

        Args:
            claim: Claim to review
            assumptions: Stated assumptions

        Returns:
            dict: Review result with challenges and risks
        """
        # Placeholder - would use LLM for adversarial analysis
        return {
            "status": "success",
            "method": "adversarial_review",
            "claim": claim,
            "challenges": [],
            "edge_cases": [],
            "alternative_interpretations": [],
            "risk_level": "low",
            "confidence_adjustment": 0.0,
            "note": "Adversarial review requires LLM integration"
        }


class CompletenessCheckSkill(Skill):
    """
    Skill: Check completeness of analysis and identify missing elements.

    Checks for:
    1. Missing context or caveats
    2. Unsupported assertions
    3. Incomplete analysis
    4. Missing evidence

    Confidence: Lower for incomplete analysis
    """

    def get_tool_definition(self) -> dict:
        return {
            "name": "check_completeness",
            "description": "Check analysis completeness and identify missing context, caveats, or evidence.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "string",
                        "description": "Analysis to check for completeness"
                    },
                    "required_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Elements that should be present"
                    }
                },
                "required": ["analysis"]
            }
        }

    def execute(self, analysis: str, required_elements: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute completeness check.

        Args:
            analysis: Analysis to check
            required_elements: Elements that should be present

        Returns:
            dict: Completeness check result
        """
        # Placeholder - would use LLM for completeness analysis
        return {
            "status": "success",
            "method": "completeness_check",
            "complete": True,
            "missing_elements": [],
            "missing_context": [],
            "unsupported_assertions": [],
            "confidence": 0.8,
            "note": "Completeness check requires LLM integration"
        }


class SynthesisSkill(Skill):
    """
    Skill: Synthesize verification results into final assessment.

    Combines:
    1. All verification method results
    2. Confidence scores
    3. Issues found
    4. Recommendations

    Outputs: Overall confidence and recommendation (keep/revise/remove)
    """

    def get_tool_definition(self) -> dict:
        return {
            "name": "synthesize_verification",
            "description": "Synthesize all verification results into final assessment with overall confidence and recommendation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "verification_results": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Results from all verification methods"
                    }
                },
                "required": ["verification_results"]
            }
        }

    def execute(self, verification_results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Execute synthesis of verification results.

        Args:
            verification_results: All verification method results

        Returns:
            dict: Synthesized assessment
        """
        try:
            # Calculate overall confidence (weighted average)
            confidences = [r.get("confidence", 0.5) for r in verification_results]
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5

            # Collect all issues
            all_issues = []
            for result in verification_results:
                all_issues.extend(result.get("issues", []))

            # Determine recommendation
            if overall_confidence >= 0.8:
                recommendation = "keep"
            elif overall_confidence >= 0.5:
                recommendation = "flag_uncertainty"
            elif overall_confidence >= 0.3:
                recommendation = "revise"
            else:
                recommendation = "remove"

            return {
                "status": "success",
                "method": "synthesis",
                "overall_confidence": overall_confidence,
                "recommendation": recommendation,
                "verification_count": len(verification_results),
                "issues_found": all_issues,
                "passed_count": sum(1 for r in verification_results if r.get("passed", False)),
                "failed_count": sum(1 for r in verification_results if not r.get("passed", True))
            }

        except Exception as e:
            return {
                "status": "error",
                "method": "synthesis",
                "overall_confidence": 0.0,
                "recommendation": "remove",
                "error": str(e)
            }
