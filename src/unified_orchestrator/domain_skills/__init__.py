"""
Domain-Specific Skills
======================

Skills specialized for different domains (NIL, Energy, Medical, etc.)

Each domain provides:
- Data fetching skills (APIs, databases, web sources)
- Validation skills (domain-specific rules and constraints)
- Verification skills (fact-checking against domain knowledge)

Available Domains:
- NIL: College player valuation (stats, market data, valuations)
- Energy: Manufacturing efficiency analysis (coming soon)
- Medical: Healthcare information verification (coming soon)
"""

from .base_domain import (
    DomainSkill,
    DataFetchSkill,
    ValidationSkill,
    DomainConfig
)

__all__ = [
    "DomainSkill",
    "DataFetchSkill",
    "ValidationSkill",
    "DomainConfig",
]
