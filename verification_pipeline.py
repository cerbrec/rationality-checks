"""
Base Verification Pipeline
Defines core data structures and LLM verification methods
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# ============================================================================
# ENUMS
# ============================================================================

class ClaimType(Enum):
    """Types of claims that can be extracted from LLM outputs"""
    FACTUAL = "factual"              # Verifiable facts
    QUANTITATIVE = "quantitative"    # Numerical/measurable claims
    CAUSAL = "causal"                # Cause-effect relationships
    LOGICAL = "logical"              # Logical inferences
    INTERPRETIVE = "interpretive"    # Subjective interpretations
    PREDICTIVE = "predictive"        # Future predictions
    ASSUMPTION = "assumption"        # Stated or implicit assumptions


class VerificationMethod(Enum):
    """Methods used to verify claims"""
    EMPIRICAL_TEST = "empirical_test"              # Logical consistency testing
    FACT_CHECK = "fact_check"                      # External fact verification
    ADVERSARIAL_REVIEW = "adversarial_review"      # Challenge the claim
    WORLD_STATE = "world_state"                    # Formal verification


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Evidence:
    """Evidence supporting or refuting a claim"""
    source: str
    content: str
    supports: bool
    confidence: float


@dataclass
class Claim:
    """A claim extracted from LLM output"""
    id: str
    text: str
    claim_type: ClaimType
    source_section: str
    dependencies: List[str] = field(default_factory=list)
    context: Dict = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of verifying a single claim"""
    claim_id: str
    method: VerificationMethod
    passed: bool
    confidence: float
    evidence: List[Evidence]
    issues_found: List[str] = field(default_factory=list)
    suggested_revision: Optional[str] = None


@dataclass
class ClaimAssessment:
    """Complete assessment of a claim"""
    claim: Claim
    verification_results: List[VerificationResult]
    overall_confidence: float
    recommendation: str  # "keep", "revise", "remove", "flag_uncertainty"
    revised_text: Optional[str] = None


@dataclass
class VerificationReport:
    """Complete verification report"""
    original_output: str
    original_query: str
    extracted_claims: List[Claim]
    assessments: List[ClaimAssessment]
    missing_elements: List[str]
    improved_output: str
    summary: str


# ============================================================================
# LLM PROVIDER INTERFACE
# ============================================================================

class LLMProvider(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt for context

        Returns:
            Generated text response
        """
        pass


# ============================================================================
# EXAMPLE LLM PROVIDERS
# ============================================================================

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Returns a mock response"""
        return '{"claims": [], "results": []}'


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate using OpenAI API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate using Anthropic API"""
        message_kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        }

        if system_prompt:
            message_kwargs["system"] = system_prompt

        response = self.client.messages.create(**message_kwargs)
        return response.content[0].text


class BedrockProvider(LLMProvider):
    """AWS Bedrock Claude API provider"""

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
        model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    ):
        """
        Initialize Bedrock provider

        Args:
            aws_access_key_id: AWS access key (defaults to env var)
            aws_secret_access_key: AWS secret key (defaults to env var)
            region_name: AWS region
            model_id: Bedrock model ID (options:
                - us.anthropic.claude-sonnet-4-20250514-v1:0 (Sonnet 4)
                - us.anthropic.claude-opus-4-1-20250805-v1:0 (Opus 4)
            )
        """
        import os

        self.model_id = model_id
        self.region_name = region_name

        # Get credentials from params or env
        self.aws_access_key_id = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')

        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError(
                "AWS credentials required. Provide via parameters or set "
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            )

        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError("boto3 required for Bedrock. Install with: pip install boto3")

        # Configure with retries
        retry_config = Config(
            retries={"max_attempts": 10, "mode": "standard"},
            read_timeout=6000,
        )

        # Create session
        session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )

        # Create bedrock-runtime client
        self.client = session.client(
            service_name='bedrock-runtime',
            config=retry_config,
            region_name=self.region_name,
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate using AWS Bedrock Converse API

        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        # Build messages
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]

        # Build request kwargs
        request_kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": 32000,
                "temperature": 0.0
            }
        }

        # Add system prompt if provided
        if system_prompt:
            request_kwargs["system"] = [{"text": system_prompt}]

        # Call Bedrock Converse API
        response = self.client.converse(**request_kwargs)

        # Extract text from response
        output = response.get("output", {})
        message = output.get("message", {})
        content_items = message.get("content", [])

        # Find text content
        for content_item in content_items:
            if "text" in content_item:
                return content_item["text"]

        raise ValueError("No text content in Bedrock response")

    @classmethod
    def from_env(cls, model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"):
        """
        Create provider from environment variables

        Reads from:
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - AWS_REGION (defaults to us-east-1)
        """
        import os
        return cls(
            region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
            model_id=model_id
        )


# ============================================================================
# BASIC VERIFICATION PIPELINE
# ============================================================================

class BasicVerificationPipeline:
    """
    Basic LLM verification pipeline without formal verification.
    This is the baseline that IntegratedVerificationPipeline improves upon.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def verify_analysis(
        self,
        original_output: str,
        original_query: str,
        enable_tool_use: bool = True
    ) -> VerificationReport:
        """
        Basic verification using only LLM-based methods.
        """
        # Simplified implementation for demonstration
        return VerificationReport(
            original_output=original_output,
            original_query=original_query,
            extracted_claims=[],
            assessments=[],
            missing_elements=[],
            improved_output=original_output,
            summary="Basic verification complete"
        )
