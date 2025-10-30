"""
Unified Intelligent Orchestrator
=================================

A unified 7-step orchestration system that integrates rationality verification
as skills/tools used during intelligent report generation, not post-processing.

This module combines:
- 7-step intelligent workflow (goal → strategy → execution → artifact)
- Rationality verification (world state, fact-checking, adversarial review)
- Domain-specific skills (NIL, Energy, Medical, etc.)

Main Components:
- IntelligentOrchestrator: Main orchestrator that coordinates 7-step workflow
- VerificationSkills: Rationality checks as reusable skills
- DomainSkills: Domain-specific capabilities (NIL player valuation, etc.)

Usage:
    from src.unified_orchestrator import IntelligentOrchestrator

    orchestrator = IntelligentOrchestrator(domain="nil")
    report = orchestrator.generate_report(
        query="Evaluate Travis Hunter's NIL market value",
        context="Colorado WR/DB, Heisman finalist 2024"
    )
"""

from .intelligent_orchestrator import IntelligentOrchestrator
from .verification_skills import (
    Skill,
    WorldStateVerificationSkill,
    EmpiricalTestingSkill,
    FactCheckingSkill,
    AdversarialReviewSkill,
    CompletenessCheckSkill,
    SynthesisSkill
)

__version__ = "0.1.0"

__all__ = [
    "IntelligentOrchestrator",
    "Skill",
    "WorldStateVerificationSkill",
    "EmpiricalTestingSkill",
    "FactCheckingSkill",
    "AdversarialReviewSkill",
    "CompletenessCheckSkill",
    "SynthesisSkill",
]
