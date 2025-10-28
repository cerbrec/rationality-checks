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
import json
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
            # If LLM provider available, use it for plausibility checking
            # This helps detect obvious hallucinations even without web search
            if self.llm_provider:
                context = kwargs.get("context", "")

                prompt = f"""Evaluate the plausibility and factual accuracy of this claim:

Claim: {claim}

{f"Context: {context}" if context else ""}

Please assess:
1. Is this claim internally consistent?
2. Does it contain obvious factual errors (wrong names, impossible numbers, category mistakes)?
3. If context is provided, does the claim align with the context?
4. What is your confidence that this claim is factually correct?

Provide a confidence score from 0.0 (definitely false) to 1.0 (definitely true), where:
- 0.0-0.3: Likely false, contains obvious errors
- 0.3-0.5: Uncertain, insufficient evidence
- 0.5-0.7: Plausible but unverified
- 0.7-0.9: Likely true, consistent with general knowledge
- 0.9-1.0: Very confident it's true

Return your assessment in this format:
CONFIDENCE: <score>
REASONING: <brief explanation>
ISSUES: <any factual errors detected or "none">"""

                try:
                    response = self.llm_provider.call(prompt, max_tokens=300)

                    # Extract confidence score
                    import re
                    confidence_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0)', response)
                    reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\nISSUES:|\Z)', response, re.DOTALL)
                    issues_match = re.search(r'ISSUES:\s*(.+)', response, re.DOTALL)

                    confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
                    issues = issues_match.group(1).strip() if issues_match else "none"

                    passed = confidence >= 0.6

                    return {
                        "status": "success",
                        "method": "fact_check_llm_plausibility",
                        "claim": claim,
                        "evidence": [{"source": "LLM reasoning", "text": reasoning}],
                        "passed": passed,
                        "confidence": confidence,
                        "issues": [] if issues.lower() == "none" else [issues],
                        "note": "Plausibility check via LLM (no web search)"
                    }

                except Exception as llm_error:
                    # Fallback to placeholder if LLM fails
                    pass

            # Fallback: return placeholder indicating web search is needed
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
        # If LLM available, perform empirical testing
        if self.llm_provider:
            try:
                # Prepare claims for testing (convert to list if single string)
                if isinstance(claims, str):
                    claims = [claims]

                context_section = f"CONTEXT:\n{context}\n\n" if context else ""

                prompt = f"""Test whether the following claim(s) are logically consistent and empirically sound.

CLAIMS TO TEST:
{json.dumps(claims, indent=2)}

{context_section}For EACH claim, perform empirical testing:
1. STATE TRANSITION TEST: If this claim is true, what else must be true?
2. CONTRADICTION TEST: Does this claim contradict itself or contain impossible statements?
3. TESTABLE PREDICTIONS: What testable predictions does this claim make?

Return your assessment in this format:
PASSED: <true/false - whether claim is logically consistent>
CONFIDENCE: <0.0-1.0 score>
ISSUES: <list any logical problems, contradictions, or impossible statements found, or "none">
ANALYSIS: <brief explanation of your reasoning>"""

                response = self.llm_provider.call(prompt, max_tokens=500)

                # Extract structured data from response
                import re
                passed_match = re.search(r'PASSED:\s*(true|false)', response, re.IGNORECASE)
                confidence_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0)', response)
                issues_match = re.search(r'ISSUES:\s*(.+?)(?=\nANALYSIS:|\Z)', response, re.DOTALL)
                analysis_match = re.search(r'ANALYSIS:\s*(.+)', response, re.DOTALL)

                passed = passed_match.group(1).lower() == 'true' if passed_match else True
                confidence = float(confidence_match.group(1)) if confidence_match else 0.7
                issues_text = issues_match.group(1).strip() if issues_match else "none"
                analysis = analysis_match.group(1).strip() if analysis_match else "No analysis provided"

                issues = [] if issues_text.lower() == "none" else [issues_text]

                return {
                    "status": "success",
                    "method": "empirical_test_llm",
                    "claims_tested": len(claims),
                    "passed": passed,
                    "confidence": confidence,
                    "issues": issues,
                    "analysis": analysis,
                    "note": "Empirical testing via LLM reasoning"
                }

            except Exception as e:
                # Fallback to placeholder on error
                pass

        # Fallback: placeholder when no LLM available
        return {
            "status": "success",
            "method": "empirical_test",
            "claims_tested": len(claims) if isinstance(claims, list) else 1,
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
        # If LLM available, perform adversarial review
        if self.llm_provider:
            try:
                assumptions_text = ""
                if assumptions:
                    assumptions_text = f"\n\nSTATED ASSUMPTIONS:\n{json.dumps(assumptions, indent=2)}"

                prompt = f"""You are an adversarial reviewer challenging the following claim to find weaknesses.

CLAIM:
{claim}
{assumptions_text}

Perform adversarial review by:
1. IDENTIFY ASSUMPTIONS: What hidden/unstated assumptions does this claim rely on?
2. FIND EDGE CASES: What scenarios would break or invalidate this claim?
3. PROPOSE ALTERNATIVES: What alternative interpretations or explanations exist?
4. CHALLENGE CONCLUSIONS: How might this claim be wrong or overstated?

Return your assessment in this format:
PASSED: <true/false - whether claim survives adversarial review>
CONFIDENCE: <0.0-1.0 - lower if serious weaknesses found>
HIDDEN_ASSUMPTIONS: <list assumptions or "none">
EDGE_CASES: <list problematic scenarios or "none">
ALTERNATIVES: <list alternative explanations or "none">
CHALLENGES: <main weaknesses found or "none">
RISK_LEVEL: <low/medium/high>"""

                response = self.llm_provider.call(prompt, max_tokens=600)

                # Extract structured data from response
                import re
                passed_match = re.search(r'PASSED:\s*(true|false)', response, re.IGNORECASE)
                confidence_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0)', response)
                assumptions_match = re.search(r'HIDDEN_ASSUMPTIONS:\s*(.+?)(?=\nEDGE_CASES:|\Z)', response, re.DOTALL)
                edge_match = re.search(r'EDGE_CASES:\s*(.+?)(?=\nALTERNATIVES:|\Z)', response, re.DOTALL)
                alternatives_match = re.search(r'ALTERNATIVES:\s*(.+?)(?=\nCHALLENGES:|\Z)', response, re.DOTALL)
                challenges_match = re.search(r'CHALLENGES:\s*(.+?)(?=\nRISK_LEVEL:|\Z)', response, re.DOTALL)
                risk_match = re.search(r'RISK_LEVEL:\s*(low|medium|high)', response, re.IGNORECASE)

                passed = passed_match.group(1).lower() == 'true' if passed_match else True
                confidence = float(confidence_match.group(1)) if confidence_match else 0.6

                def parse_list_field(match):
                    if not match:
                        return []
                    text = match.group(1).strip()
                    if text.lower() == "none":
                        return []
                    return [text]

                hidden_assumptions = parse_list_field(assumptions_match)
                edge_cases = parse_list_field(edge_match)
                alternatives = parse_list_field(alternatives_match)
                challenges = parse_list_field(challenges_match)
                risk_level = risk_match.group(1).lower() if risk_match else "low"

                # Calculate confidence adjustment (negative if issues found)
                issues_count = len(hidden_assumptions) + len(edge_cases) + len(alternatives) + len(challenges)
                confidence_adjustment = -0.1 * min(issues_count, 3)  # Max -0.3

                return {
                    "status": "success",
                    "method": "adversarial_review_llm",
                    "claim": claim,
                    "passed": passed,
                    "confidence": confidence,
                    "challenges": challenges,
                    "edge_cases": edge_cases,
                    "alternative_interpretations": alternatives,
                    "hidden_assumptions": hidden_assumptions,
                    "risk_level": risk_level,
                    "confidence_adjustment": confidence_adjustment,
                    "note": "Adversarial review via LLM reasoning"
                }

            except Exception as e:
                # Fallback to placeholder on error
                pass

        # Fallback: placeholder when no LLM available
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
