"""
Dynamic Claim Type Discovery and Enhanced Verification

This module provides a GENERAL framework for:
1. LLM-powered discovery of claim patterns from any document type
2. Entity extraction for targeted verification
3. Context-aware search query generation
4. Time-sensitivity detection

Rather than hardcoded patterns, this uses the LLM to understand the document
and identify what types of claims need verification.
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class TimeSensitivity(Enum):
    """How frequently this type of claim changes"""
    HIGH = "high"      # Changes weekly/monthly (rosters, prices, employment)
    MEDIUM = "medium"  # Changes quarterly/yearly (partnerships, products)
    LOW = "low"        # Relatively stable (company history, locations)


@dataclass
class DiscoveredClaimPattern:
    """A claim pattern discovered by the LLM"""
    name: str
    description: str
    example_claims: List[str]
    key_entities: List[str]  # What entities to extract (names, dates, numbers, etc.)
    verification_strategy: str  # How to verify this type of claim
    search_query_templates: List[str]  # LLM-generated search templates
    time_sensitivity: str  # "high", "medium", "low"
    severity: str  # "CRITICAL", "MAJOR", "MINOR"
    why_matters: str  # Impact if claim is wrong


# ============================================================================
# CORE: LLM-POWERED CLAIM PATTERN DISCOVERY
# ============================================================================

def discover_claim_patterns_from_document(
    document: str,
    llm_provider,
    document_domain: str = "auto"
) -> List[DiscoveredClaimPattern]:
    """
    Use LLM to analyze document and discover verifiable claim patterns.

    This is the PRIMARY method - it adapts to ANY document type.

    Args:
        document: The document to analyze
        llm_provider: LLM provider for analysis
        document_domain: Domain hint ("auto", "business", "medical", etc.)

    Returns:
        List of discovered claim patterns
    """

    # Prepare analysis prompt
    prompt = f"""You are analyzing a document to identify verifiable claim patterns for fact-checking.

DOCUMENT (first 3000 characters):
{document[:3000]}

Your task is to identify CLAIM PATTERNS that appear in this document and need verification.

For each pattern type, provide:
1. **Pattern Name**: Short identifier (e.g., "person_current_role", "financial_projection")
2. **Description**: What this claim pattern represents
3. **Example Claims**: 2-3 actual examples from the document
4. **Key Entities**: What information to extract (person names, companies, dates, numbers, etc.)
5. **Verification Strategy**: How to verify this type of claim
6. **Search Query Templates**: 3-5 search queries to verify (use {{entity}} placeholders)
7. **Time Sensitivity**: How often this data changes (high/medium/low)
8. **Severity**: Impact if wrong (CRITICAL/MAJOR/MINOR)
9. **Why It Matters**: What could go wrong if this claim is false

Focus on:
- Claims with specific people, companies, products, or entities that can become outdated
- Claims with numbers, statistics, or financial data that can be verified
- Claims about current status, relationships, or partnerships
- Claims about capabilities, availability, or specifications

Return ONLY valid JSON with this structure:
{{
  "patterns": [
    {{
      "name": "pattern_identifier",
      "description": "what this pattern represents",
      "example_claims": ["claim 1", "claim 2"],
      "key_entities": ["entity_type_1", "entity_type_2"],
      "verification_strategy": "how to verify",
      "search_query_templates": ["{{entity1}} {{entity2}} verification query"],
      "time_sensitivity": "high|medium|low",
      "severity": "CRITICAL|MAJOR|MINOR",
      "why_matters": "impact if false"
    }}
  ]
}}

Focus on patterns that are TIME-SENSITIVE or have high ERROR RISK."""

    # Get LLM response
    response = llm_provider.generate(
        prompt,
        system_prompt="You are an expert document analyst identifying verifiable claims. Return ONLY valid JSON."
    )

    # Parse response
    try:
        # Extract JSON from response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)

            patterns = []
            for p in parsed.get("patterns", []):
                patterns.append(DiscoveredClaimPattern(
                    name=p.get("name", "unknown"),
                    description=p.get("description", ""),
                    example_claims=p.get("example_claims", []),
                    key_entities=p.get("key_entities", []),
                    verification_strategy=p.get("verification_strategy", ""),
                    search_query_templates=p.get("search_query_templates", []),
                    time_sensitivity=p.get("time_sensitivity", "medium"),
                    severity=p.get("severity", "MINOR"),
                    why_matters=p.get("why_matters", "")
                ))

            return patterns

    except Exception as e:
        print(f"[Dynamic Claims] Failed to parse LLM response: {e}")
        return []


# ============================================================================
# ENTITY EXTRACTION (General Purpose)
# ============================================================================

def extract_entities_from_claim(
    claim_text: str,
    entity_types: List[str],
    llm_provider
) -> Dict[str, str]:
    """
    Use LLM to extract specific entities from a claim.

    This is GENERAL - works for any entity type (people, companies, dates, etc.)

    Args:
        claim_text: The claim to extract from
        entity_types: Types of entities to extract (e.g., ["person_name", "company", "year"])
        llm_provider: LLM provider

    Returns:
        Dictionary mapping entity type to extracted value
    """

    prompt = f"""Extract the following entities from this claim:

CLAIM: "{claim_text}"

ENTITIES TO EXTRACT: {', '.join(entity_types)}

Return as JSON: {{"entity_type": "extracted_value", ...}}

If an entity is not present, use null.
Return ONLY the JSON object."""

    response = llm_provider.generate(prompt, system_prompt="Extract entities precisely. Return only JSON.")

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            entities = json.loads(json_str)
            # Filter out null values
            return {k: v for k, v in entities.items() if v is not None}
    except:
        return {}


# ============================================================================
# CONTEXT EXTRACTION (General Purpose)
# ============================================================================

def extract_document_context(document: str, llm_provider) -> Dict[str, str]:
    """
    Use LLM to extract contextual information from document.

    GENERAL - adapts to any document type.

    Args:
        document: The document
        llm_provider: LLM provider

    Returns:
        Context dictionary (domain, primary_subjects, time_period, etc.)
    """

    prompt = f"""Analyze this document and extract key contextual information:

DOCUMENT (first 2000 characters):
{document[:2000]}

Extract:
1. **Domain**: What field is this? (sports, business, medical, technology, etc.)
2. **Primary Subjects**: Main entities discussed (teams, companies, people, products)
3. **Time Period**: What time period is referenced? (specific year, season, current, future)
4. **Geographic Focus**: Any specific locations mentioned?
5. **Document Type**: What kind of document is this? (analysis, report, proposal, etc.)

Return as JSON:
{{
  "domain": "field",
  "primary_subjects": ["subject1", "subject2"],
  "time_period": "when",
  "geographic_focus": "where",
  "document_type": "type"
}}

Return ONLY the JSON object."""

    response = llm_provider.generate(prompt, system_prompt="Extract context. Return only JSON.")

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
    except:
        return {}


# ============================================================================
# SEARCH QUERY GENERATION (Context-Aware)
# ============================================================================

def generate_verification_queries(
    claim_text: str,
    entities: Dict[str, str],
    query_templates: List[str],
    document_context: Dict[str, str]
) -> List[str]:
    """
    Generate search queries from templates and entities.

    GENERAL - fills in templates with extracted entities and context.

    Args:
        claim_text: The original claim
        entities: Extracted entities
        query_templates: Templates from discovered pattern
        document_context: Document context

    Returns:
        List of search query strings
    """

    queries = []

    # Combine entities and context for template filling
    all_data = {**entities, **document_context}

    for template in query_templates:
        try:
            # Try to fill template with available data
            query = template
            for key, value in all_data.items():
                placeholder = "{" + key + "}"
                if placeholder in query:
                    query = query.replace(placeholder, str(value))

            # Only add if we successfully filled required placeholders
            if "{" not in query or "}" not in query:
                queries.append(query)

        except Exception:
            continue

    return queries


def generate_negative_evidence_queries(
    claim_text: str,
    entities: Dict[str, str],
    document_context: Dict[str, str],
    llm_provider
) -> List[str]:
    """
    Use LLM to generate queries that would CONTRADICT the claim.

    GENERAL - works for any claim type.

    Args:
        claim_text: The claim to verify
        entities: Extracted entities
        document_context: Context
        llm_provider: LLM provider

    Returns:
        List of negative evidence search queries
    """

    prompt = f"""Generate search queries to find evidence that would CONTRADICT this claim:

CLAIM: "{claim_text}"

ENTITIES: {json.dumps(entities)}
CONTEXT: {json.dumps(document_context)}

Generate 3-5 search queries that would find:
- Evidence the claim is outdated or no longer true
- Evidence of changes that invalidate the claim
- Evidence contradicting the claim's assertions

Return as JSON array: ["query 1", "query 2", ...]
Return ONLY the JSON array."""

    response = llm_provider.generate(prompt, system_prompt="Generate contradiction-seeking queries. Return only JSON array.")

    try:
        json_start = response.find("[")
        json_end = response.rfind("]") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
    except:
        return []


# ============================================================================
# THREE-TIER VERIFICATION WORKFLOW
# ============================================================================

@dataclass
class VerificationPlan:
    """Plan for verifying a specific claim"""
    claim_text: str
    pattern: DiscoveredClaimPattern
    entities: Dict[str, str]
    tier1_queries: List[str]  # Direct verification
    tier2_queries: List[str]  # Cross-reference
    tier3_queries: List[str]  # Negative evidence
    severity_multiplier: float


def create_verification_plan(
    claim_text: str,
    pattern: DiscoveredClaimPattern,
    entities: Dict[str, str],
    document_context: Dict[str, str],
    llm_provider
) -> VerificationPlan:
    """
    Create a complete verification plan for a claim.

    GENERAL - adapts verification strategy to claim type.
    """

    # Tier 1: Direct verification using pattern templates
    tier1_queries = generate_verification_queries(
        claim_text,
        entities,
        pattern.search_query_templates,
        document_context
    )

    # Tier 2: Cross-reference queries (generated by LLM)
    tier2_prompt = f"""Generate 2-3 cross-reference search queries to verify this claim from different angles:

CLAIM: "{claim_text}"
ENTITIES: {json.dumps(entities)}

Return as JSON array: ["query 1", "query 2"]
Return ONLY the JSON array."""

    try:
        tier2_response = llm_provider.generate(tier2_prompt, system_prompt="Return only JSON array.")
        json_start = tier2_response.find("[")
        json_end = tier2_response.rfind("]") + 1
        if json_start != -1:
            tier2_queries = json.loads(tier2_response[json_start:json_end])
        else:
            tier2_queries = []
    except:
        tier2_queries = []

    # Tier 3: Negative evidence queries
    tier3_queries = generate_negative_evidence_queries(
        claim_text,
        entities,
        document_context,
        llm_provider
    )

    # Calculate severity multiplier based on time sensitivity
    severity_multiplier = {
        "high": 2.0,
        "medium": 1.5,
        "low": 1.0
    }.get(pattern.time_sensitivity, 1.0)

    return VerificationPlan(
        claim_text=claim_text,
        pattern=pattern,
        entities=entities,
        tier1_queries=tier1_queries[:4],  # Limit queries
        tier2_queries=tier2_queries[:3],
        tier3_queries=tier3_queries[:3],
        severity_multiplier=severity_multiplier
    )


# ============================================================================
# INTEGRATION WITH EXISTING CLAIM SYSTEM
# ============================================================================

def extract_claims_with_patterns(
    document: str,
    discovered_patterns: List[DiscoveredClaimPattern],
    document_context: Dict[str, str],
    llm_provider
) -> List[Dict[str, Any]]:
    """
    Extract specific claims from document based on discovered patterns.

    GENERAL - uses LLM to find claims matching each pattern.

    Args:
        document: The document
        discovered_patterns: Patterns found by discover_claim_patterns_from_document
        document_context: Context from extract_document_context
        llm_provider: LLM provider

    Returns:
        List of extracted claims with verification plans
    """

    extracted_claims = []

    for i, pattern in enumerate(discovered_patterns[:5], 1):  # Limit to top 5 patterns for performance
        print(f"[Dynamic Claims] Processing pattern {i}/5: {pattern.name}...")
        # Ask LLM to find all claims matching this pattern
        prompt = f"""Find ALL claims in this document that match this pattern:

PATTERN: {pattern.description}
EXAMPLE CLAIMS: {', '.join(pattern.example_claims[:2])}

DOCUMENT:
{document}

Return a JSON array of matching claim texts:
["claim text 1", "claim text 2", ...]

Return ONLY the JSON array."""

        try:
            response = llm_provider.generate(prompt, system_prompt="Find matching claims. Return only JSON array.")
            json_start = response.find("[")
            json_end = response.rfind("]") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                matching_claims = json.loads(json_str)

                # For each matching claim, extract entities and create plan
                print(f"[Dynamic Claims]   Found {len(matching_claims)} matching claims, processing top {min(3, len(matching_claims))}...")
                for claim_text in matching_claims[:3]:  # Limit to top 3 per pattern for performance
                    # Extract entities
                    entities = extract_entities_from_claim(
                        claim_text,
                        pattern.key_entities,
                        llm_provider
                    )

                    if entities:
                        # Create verification plan
                        plan = create_verification_plan(
                            claim_text,
                            pattern,
                            entities,
                            document_context,
                            llm_provider
                        )

                        extracted_claims.append({
                            "claim_text": claim_text,
                            "pattern_name": pattern.name,
                            "pattern": pattern,
                            "entities": entities,
                            "verification_plan": plan,
                            "severity": pattern.severity,
                            "time_sensitivity": pattern.time_sensitivity
                        })

        except Exception as e:
            print(f"[Dynamic Claims] Error extracting claims for pattern {pattern.name}: {e}")
            continue

    return extracted_claims


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def discover_and_extract_claims(
    document: str,
    llm_provider,
    enable_dynamic_discovery: bool = True
) -> tuple[List[DiscoveredClaimPattern], List[Dict[str, Any]], Dict[str, str]]:
    """
    Complete workflow: discover patterns, extract claims, create verification plans.

    GENERAL - works for ANY document type.

    Args:
        document: Document to analyze
        llm_provider: LLM provider
        enable_dynamic_discovery: Whether to use LLM discovery (vs hardcoded patterns)

    Returns:
        Tuple of (discovered_patterns, extracted_claims, document_context)
    """

    print("[Dynamic Claims] Extracting document context...")
    document_context = extract_document_context(document, llm_provider)
    print(f"[Dynamic Claims] Context: {document_context}")

    if enable_dynamic_discovery:
        print("[Dynamic Claims] Discovering claim patterns...")
        discovered_patterns = discover_claim_patterns_from_document(
            document,
            llm_provider,
            document_context.get("domain", "auto")
        )
        print(f"[Dynamic Claims] Discovered {len(discovered_patterns)} patterns")
    else:
        discovered_patterns = []

    print("[Dynamic Claims] Extracting claims based on patterns...")
    extracted_claims = extract_claims_with_patterns(
        document,
        discovered_patterns,
        document_context,
        llm_provider
    )
    print(f"[Dynamic Claims] Extracted {len(extracted_claims)} verifiable claims")

    return discovered_patterns, extracted_claims, document_context
