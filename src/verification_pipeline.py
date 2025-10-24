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

        # GPT-5 only supports default temperature of 1.0
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if not self.model.startswith("gpt-5"):
            kwargs["temperature"] = 0.0

        response = self.client.chat.completions.create(**kwargs)
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
        model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    ):
        """
        Initialize Bedrock provider

        Args:
            aws_access_key_id: AWS access key (defaults to env var)
            aws_secret_access_key: AWS secret key (defaults to env var)
            region_name: AWS region
            model_id: Bedrock inference profile ID (options:
                - us.anthropic.claude-sonnet-4-5-20250929-v1:0 (Sonnet 4.5 - Latest)
                - us.anthropic.claude-opus-4-1-20250805-v1:0 (Opus 4.1 - Most capable)
                - us.anthropic.claude-3-7-sonnet-20250219-v1:0 (Sonnet 3.7 - Reasoning)
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
        # Determine max tokens based on model
        # Claude models support 32k, Llama/Nova models support 8k
        if 'claude' in self.model_id.lower():
            max_tokens = 32000
        elif 'llama' in self.model_id.lower() or 'nova' in self.model_id.lower():
            max_tokens = 8192
        else:
            max_tokens = 4096  # Conservative default

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
                "maxTokens": max_tokens,
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

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        system_prompt: Optional[str] = None,
        max_tool_uses: int = 10
    ) -> tuple[str, List[Dict]]:
        """
        Generate using AWS Bedrock with tool use support

        Args:
            prompt: The prompt to send
            tools: List of tool definitions
            system_prompt: Optional system prompt
            max_tool_uses: Maximum number of tool use iterations

        Returns:
            Tuple of (final_text_response, list_of_tool_calls_made)
        """
        # Determine max tokens
        if 'claude' in self.model_id.lower():
            max_tokens = 32000
        else:
            max_tokens = 8192

        # Initialize conversation
        conversation_history = [{
            "role": "user",
            "content": [{"text": prompt}]
        }]

        tool_calls_made = []
        tool_use_count = 0

        # Import web search tool
        from .web_search import get_web_search_tool
        web_search = get_web_search_tool()

        while tool_use_count < max_tool_uses:
            # Build request
            request_kwargs = {
                "modelId": self.model_id,
                "messages": conversation_history,
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": 0.0
                },
                "toolConfig": {"tools": tools}
            }

            if system_prompt:
                request_kwargs["system"] = [{"text": system_prompt}]

            # Call Bedrock
            response = self.client.converse(**request_kwargs)

            output = response.get("output", {})
            message = output.get("message", {})
            content_items = message.get("content", [])

            # Check for tool use
            tool_results = []
            has_final_answer = False

            for content_item in content_items:
                if "toolUse" in content_item:
                    tool_use = content_item["toolUse"]
                    tool_name = tool_use.get("name")
                    tool_input = tool_use.get("input", {})
                    tool_use_id = tool_use.get("toolUseId")

                    # Handle web search
                    if tool_name == "web_search":
                        search_result = web_search.execute_from_tool_use(tool_input)
                        tool_calls_made.append({
                            "tool": "web_search",
                            "input": tool_input,
                            "result": search_result
                        })

                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": search_result}]
                            }
                        })
                    else:
                        # Unknown tool
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": f"Error: Unknown tool '{tool_name}'"}]
                            }
                        })

                elif "text" in content_item:
                    has_final_answer = True

            # Add assistant message to history
            conversation_history.append(message)

            # If there were tool calls, add results and continue
            if tool_results:
                conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })
                tool_use_count += 1
                has_final_answer = False

            # If we have a final answer, break
            if has_final_answer:
                break

        # Extract final text response
        output = response.get("output", {})
        message = output.get("message", {})
        for content_item in message.get("content", []):
            if "text" in content_item:
                return content_item["text"], tool_calls_made

        # If we reached max iterations without a final text response,
        # return the last assistant message or a summary
        print(f"  [Bedrock] Warning: Reached max tool uses ({tool_use_count}) without final text response")
        return f"Verification completed after {tool_use_count} tool uses. See tool call results for evidence.", tool_calls_made

    @classmethod
    def from_env(cls, model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"):
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


class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key
            model: Gemini model to use (options:
                - gemini-2.0-flash (Gemini 2.0 Flash - Fast, stable)
                - gemini-2.5-flash (Gemini 2.5 Flash)
                - gemini-2.5-pro (Gemini 2.5 Pro - Most capable)
                - gemini-2.0-pro-exp (Gemini 2.0 Pro Experimental)
            )
        """
        self.api_key = api_key
        self.model = model
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate using Gemini API with retry on timeout"""
        import time
        from google.api_core import exceptions as google_exceptions

        # Combine system prompt with user prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Generate with temperature=0 for consistency
        generation_config = {
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                return response.text
            except google_exceptions.DeadlineExceeded as e:
                if attempt == max_retries - 1:
                    print(f"\n❌ Gemini API timeout after {max_retries} attempts")
                    raise
                delay = base_delay * (2 ** attempt)
                print(f"  ⚠️  Gemini timeout (attempt {attempt + 1}), retrying in {delay}s...")
                time.sleep(delay)
            except google_exceptions.ResourceExhausted as e:
                # Rate limit or quota exceeded
                if attempt == max_retries - 1:
                    print(f"\n❌ Gemini API resource exhausted after {max_retries} attempts")
                    raise
                delay = base_delay * (2 ** attempt) * 2  # Longer delay for rate limits
                print(f"  ⚠️  Gemini rate limit (attempt {attempt + 1}), retrying in {delay}s...")
                time.sleep(delay)


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
