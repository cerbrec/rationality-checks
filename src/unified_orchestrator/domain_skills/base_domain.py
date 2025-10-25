"""
Base Domain Skill Classes
=========================

Abstract base classes for domain-specific skills.
Each domain (NIL, Energy, Medical) extends these classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DomainConfig:
    """Configuration for a specific domain"""
    name: str
    data_sources: Dict[str, List[str]]
    verification_thresholds: Dict[str, float]
    required_verifications: List[str]
    api_keys: Optional[Dict[str, str]] = None


class DomainSkill(ABC):
    """
    Abstract base class for domain-specific skills.

    Domain skills provide specialized capabilities for specific domains
    like NIL player valuation, energy efficiency analysis, or medical
    information verification.
    """

    def __init__(self, config: DomainConfig):
        self.config = config

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
        Execute the domain skill.

        Returns:
            dict: Result with status, data, confidence, and metadata
        """
        pass

    def validate_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate that the result meets domain requirements.

        Args:
            result: Execution result to validate

        Returns:
            bool: True if result is valid
        """
        return (
            "status" in result and
            "data" in result and
            "confidence" in result and
            0.0 <= result["confidence"] <= 1.0
        )


class DataFetchSkill(DomainSkill):
    """
    Abstract base class for data fetching skills.

    Data fetching skills retrieve information from external sources
    like APIs, databases, web pages, etc.
    """

    @abstractmethod
    def fetch_data(self, query: str, source: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch data from external sources.

        Args:
            query: What data to fetch
            source: Optional specific source to query

        Returns:
            dict: Fetched data with metadata
        """
        pass

    def execute(self, query: str, source: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Execute data fetch operation"""
        try:
            data = self.fetch_data(query, source)
            return {
                "status": "success",
                "data": data,
                "confidence": 1.0 if data else 0.0,
                "source": source or "default",
                "query": query
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "confidence": 0.0,
                "error": str(e),
                "query": query
            }


class ValidationSkill(DomainSkill):
    """
    Abstract base class for validation skills.

    Validation skills check data against domain-specific rules,
    constraints, and standards.
    """

    @abstractmethod
    def validate(self, data: Dict[str, Any], rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate data against domain rules.

        Args:
            data: Data to validate
            rules: Optional specific rules to apply

        Returns:
            dict: Validation result with passed/failed status and issues
        """
        pass

    def execute(self, data: Dict[str, Any], rules: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Execute validation operation"""
        try:
            result = self.validate(data, rules)
            return {
                "status": "success",
                "data": result,
                "confidence": result.get("confidence", 0.5),
                "passed": result.get("passed", False),
                "issues": result.get("issues", [])
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "confidence": 0.0,
                "passed": False,
                "error": str(e),
                "issues": [str(e)]
            }


class VerificationSkill(DomainSkill):
    """
    Abstract base class for verification skills.

    Verification skills check claims against external evidence,
    fact-check statements, and validate assertions.
    """

    @abstractmethod
    def verify_claim(self, claim: str, evidence: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Verify a claim against evidence.

        Args:
            claim: Claim to verify
            evidence: Optional pre-fetched evidence

        Returns:
            dict: Verification result with confidence and supporting evidence
        """
        pass

    def execute(self, claim: str, evidence: Optional[List[Dict]] = None, **kwargs) -> Dict[str, Any]:
        """Execute verification operation"""
        try:
            result = self.verify_claim(claim, evidence)
            return {
                "status": "success",
                "data": result,
                "confidence": result.get("confidence", 0.5),
                "verified": result.get("verified", False),
                "evidence": result.get("evidence", []),
                "claim": claim
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "confidence": 0.0,
                "verified": False,
                "error": str(e),
                "claim": claim
            }
