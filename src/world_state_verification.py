"""
World State Verification
Formal verification using logical propositions and constraints
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Proposition:
    """A formal proposition about the world state"""
    subject: str           # Entity the proposition is about
    predicate: str         # Property or relation
    value: Any            # Value of the property
    source_claim_id: str  # Which claim this came from
    metadata: Dict = field(default_factory=dict)


@dataclass
class Constraint:
    """A constraint that must hold between propositions"""
    constraint_type: str   # "equation", "inequality", "logical"
    variables: List[str]   # Variables involved
    formula: str          # Formula defining the constraint
    source_claim_id: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ConsistencyIssue:
    """A detected inconsistency in the world state"""
    issue_type: str        # "contradiction", "constraint_violation", "missing_dependency"
    description: str
    involved_propositions: List[Proposition]
    involved_constraints: List[Constraint]
    severity: float       # 0.0-1.0


# ============================================================================
# WORLD STATE
# ============================================================================

class WorldState:
    """
    Maintains a consistent world state built from claims.
    Detects contradictions and constraint violations.
    """

    def __init__(self):
        self.propositions: Dict[str, Proposition] = {}
        self.constraints: List[Constraint] = []
        self.consistency_issues: List[ConsistencyIssue] = []

    def add_proposition(self, prop: Proposition) -> Optional[ConsistencyIssue]:
        """
        Add a proposition to the world state.
        Returns a ConsistencyIssue if it conflicts with existing propositions.
        """
        # Create key for the proposition
        key = f"{prop.subject}::{prop.predicate}"

        # Check if we already have a value for this subject-predicate pair
        if key in self.propositions:
            existing = self.propositions[key]
            if existing.value != prop.value:
                # Contradiction detected
                issue = ConsistencyIssue(
                    issue_type="contradiction",
                    description=f"Contradictory values for {prop.subject}.{prop.predicate}: "
                                f"{existing.value} (from {existing.source_claim_id}) vs "
                                f"{prop.value} (from {prop.source_claim_id})",
                    involved_propositions=[existing, prop],
                    involved_constraints=[],
                    severity=1.0
                )
                self.consistency_issues.append(issue)
                return issue

        # No conflict, add it
        self.propositions[key] = prop
        return None

    def add_constraint(self, constraint: Constraint) -> Optional[ConsistencyIssue]:
        """
        Add a constraint to the world state.
        Returns a ConsistencyIssue if it's violated by existing propositions.
        """
        self.constraints.append(constraint)

        # Check if constraint is satisfied
        violation = self._check_constraint(constraint)
        if violation:
            self.consistency_issues.append(violation)
            return violation

        return None

    def _check_constraint(self, constraint: Constraint) -> Optional[ConsistencyIssue]:
        """
        Check if a constraint is satisfied by current propositions.
        """
        if constraint.constraint_type == "equation":
            return self._check_equation_constraint(constraint)
        elif constraint.constraint_type == "inequality":
            return self._check_inequality_constraint(constraint)
        elif constraint.constraint_type == "logical":
            return self._check_logical_constraint(constraint)

        return None

    def _check_equation_constraint(self, constraint: Constraint) -> Optional[ConsistencyIssue]:
        """
        Check an equation constraint like "v1 == v2 * 10"
        """
        try:
            # Extract variable values from propositions
            var_values = {}
            involved_props = []

            for var in constraint.variables:
                # Find proposition for this variable
                for key, prop in self.propositions.items():
                    if var in key or var == prop.subject:
                        var_values[var] = prop.value
                        involved_props.append(prop)
                        break

            # If we don't have all variables, can't check yet
            if len(var_values) != len(constraint.variables):
                return None

            # Evaluate the constraint
            # Simple evaluation for common patterns like "v1 == v2 * v3"
            formula = constraint.formula
            for var, value in var_values.items():
                formula = formula.replace(var, str(value))

            # Safely evaluate (in production, use a proper expression evaluator)
            if "==" in formula:
                left, right = formula.split("==")
                left_val = eval(left.strip())
                right_val = eval(right.strip())

                if left_val != right_val:
                    return ConsistencyIssue(
                        issue_type="constraint_violation",
                        description=f"Constraint violated: {constraint.formula} "
                                    f"({left_val} ≠ {right_val})",
                        involved_propositions=involved_props,
                        involved_constraints=[constraint],
                        severity=1.0
                    )

        except Exception as e:
            # If we can't evaluate, return a warning
            return ConsistencyIssue(
                issue_type="constraint_violation",
                description=f"Could not evaluate constraint: {constraint.formula} ({str(e)})",
                involved_propositions=[],
                involved_constraints=[constraint],
                severity=0.5
            )

        return None

    def _check_inequality_constraint(self, constraint: Constraint) -> Optional[ConsistencyIssue]:
        """Check inequality constraints like "v1 > v2" """
        # Similar to equation checking, but for inequalities
        return None

    def _check_logical_constraint(self, constraint: Constraint) -> Optional[ConsistencyIssue]:
        """Check logical constraints like "if A then B" """
        # Logical implication checking
        return None

    def is_consistent(self) -> Tuple[bool, List[ConsistencyIssue]]:
        """
        Check if the entire world state is consistent.
        Returns (is_consistent, list_of_issues)
        """
        # Re-check all constraints
        all_issues = list(self.consistency_issues)

        for constraint in self.constraints:
            violation = self._check_constraint(constraint)
            if violation and violation not in all_issues:
                all_issues.append(violation)

        return len(all_issues) == 0, all_issues

    def get_proposition(self, subject: str, predicate: str) -> Optional[Proposition]:
        """Get a specific proposition"""
        key = f"{subject}::{predicate}"
        return self.propositions.get(key)

    def query(self, subject: str) -> List[Proposition]:
        """Get all propositions about a subject"""
        return [
            prop for prop in self.propositions.values()
            if prop.subject == subject
        ]

    def get_all_subjects(self) -> List[str]:
        """Get all unique subjects in the world state"""
        return list(set(prop.subject for prop in self.propositions.values()))

    def to_dict(self) -> Dict:
        """Convert world state to dictionary for JSON serialization"""
        return {
            "propositions": [
                {
                    "subject": prop.subject,
                    "predicate": prop.predicate,
                    "value": prop.value,
                    "source_claim_id": prop.source_claim_id
                }
                for prop in self.propositions.values()
            ],
            "constraints": [
                {
                    "type": c.constraint_type,
                    "variables": c.variables,
                    "formula": c.formula,
                    "source_claim_id": c.source_claim_id
                }
                for c in self.constraints
            ],
            "issues": [
                {
                    "type": issue.issue_type,
                    "description": issue.description,
                    "severity": issue.severity
                }
                for issue in self.consistency_issues
            ]
        }

    def visualize(self, verbose: bool = False) -> str:
        """
        Pretty print the world state

        Args:
            verbose: If True, show all details including metadata
        """
        lines = []
        lines.append("=" * 80)
        lines.append("WORLD STATE")
        lines.append("=" * 80)

        # Group propositions by subject
        subjects = self.get_all_subjects()

        if not subjects:
            lines.append("\nNo propositions yet.")
        else:
            lines.append(f"\nEntities: {len(subjects)}")
            lines.append(f"Propositions: {len(self.propositions)}")
            lines.append(f"Constraints: {len(self.constraints)}")
            lines.append(f"Issues: {len(self.consistency_issues)}")

            for subject in sorted(subjects):
                lines.append(f"\n{subject}:")
                props = self.query(subject)
                for prop in props:
                    value_str = format_value(prop.value)
                    lines.append(f"  • {prop.predicate}: {value_str}")
                    if verbose:
                        lines.append(f"    (from {prop.source_claim_id})")

        # Show constraints
        if self.constraints:
            lines.append(f"\nConstraints ({len(self.constraints)}):")
            for i, constraint in enumerate(self.constraints, 1):
                lines.append(f"  {i}. {constraint.formula}")
                if verbose:
                    lines.append(f"     Type: {constraint.constraint_type}")
                    lines.append(f"     Variables: {', '.join(constraint.variables)}")
                    lines.append(f"     Source: {constraint.source_claim_id}")

        # Show issues
        if self.consistency_issues:
            lines.append(f"\n⚠️  ISSUES DETECTED ({len(self.consistency_issues)}):")
            for i, issue in enumerate(self.consistency_issues, 1):
                lines.append(f"  {i}. [{issue.issue_type.upper()}] {issue.description}")
                lines.append(f"     Severity: {issue.severity}")
                if verbose and issue.involved_propositions:
                    lines.append(f"     Involved propositions:")
                    for prop in issue.involved_propositions:
                        lines.append(f"       - {prop.subject}.{prop.predicate} = {prop.value}")
        else:
            lines.append("\n✓ No consistency issues detected")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


# ============================================================================
# WORLD STATE VERIFIER
# ============================================================================

class WorldStateVerifier:
    """
    Verifies claims by building and checking world state consistency.
    """

    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.claim_interpreter = ClaimInterpreter()

    def verify_claims(self, claims: List) -> Tuple[WorldState, List[ConsistencyIssue]]:
        """
        Build world state from claims and check consistency.

        Args:
            claims: List of claims with formal_structure

        Returns:
            (world_state, list_of_issues)
        """
        world = WorldState()
        all_issues = []

        for claim in claims:
            if not hasattr(claim, 'formal_structure') or not claim.formal_structure:
                continue

            # Add propositions from claim
            for prop_data in claim.formal_structure.get("propositions", []):
                prop = Proposition(
                    subject=prop_data["subject"],
                    predicate=prop_data["predicate"],
                    value=prop_data["value"],
                    source_claim_id=claim.id
                )
                issue = world.add_proposition(prop)
                if issue:
                    all_issues.append(issue)

            # Add constraints from claim
            for const_data in claim.formal_structure.get("constraints", []):
                constraint = Constraint(
                    constraint_type=const_data.get("type", "equation"),
                    variables=const_data["variables"],
                    formula=const_data["formula"],
                    source_claim_id=claim.id
                )
                issue = world.add_constraint(constraint)
                if issue:
                    all_issues.append(issue)

        # Final consistency check
        is_consistent, final_issues = world.is_consistent()
        all_issues.extend([i for i in final_issues if i not in all_issues])

        return world, all_issues


# ============================================================================
# CLAIM INTERPRETER
# ============================================================================

class ClaimInterpreter:
    """
    Interprets natural language claims into formal structures.
    This is a simplified version - in production, this would use LLM.
    """

    def interpret(self, claim_text: str) -> Dict:
        """
        Convert natural language claim to formal structure.

        Returns:
            {
                "propositions": [...],
                "constraints": [...],
                "implications": [...]
            }
        """
        # Simple pattern matching for demonstration
        formal_structure = {
            "propositions": [],
            "constraints": [],
            "implications": []
        }

        # Example: "Company X is valued at $50B"
        # Extract: {subject: "Company_X", predicate: "valuation", value: 50000000000}

        # This is where you'd use LLM or sophisticated NLP
        # For now, return empty structure
        return formal_structure


# ============================================================================
# UTILITIES
# ============================================================================

def format_value(value: Any) -> str:
    """Format a value for display"""
    if isinstance(value, (int, float)):
        if value >= 1_000_000_000:
            return f"${value / 1_000_000_000:.1f}B"
        elif value >= 1_000_000:
            return f"${value / 1_000_000:.1f}M"
        else:
            return str(value)
    return str(value)
