"""
Data models for stateRuleEngine rules.

These mirror the rule types in indiCompRules.hpp 
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RuleType(Enum):
    """Rule types matching indiCompRules.hpp ruleType names."""
    numVal = "numVal"
    txtVal = "txtVal"
    swVal = "swVal"
    timeDiff = "timeDiff"
    elCompNum = "elCompNum"
    elCompTxt = "elCompTxt"
    elCompSw = "elCompSw"
    ruleComp = "ruleComp"


class Comparison(Enum):
    """Comparison operators matching indiCompRules.hpp ruleComparison."""
    Eq = "Eq"
    Neq = "Neq"
    Lt = "Lt"
    Gt = "Gt"
    LtEq = "LtEq"
    GtEq = "GtEq"
    And = "And"
    Nand = "Nand"
    Or = "Or"
    Nor = "Nor"
    Imply = "Imply"
    Nimply = "Nimply"
    Xor = "Xor"
    Xnor = "Xnor"


class Priority(Enum):
    """Reporting priorities matching indiCompRules.hpp rulePriority."""
    none = "none"
    info = "info"
    caution = "caution"
    warning = "warning"
    alert = "alert"


# Valid comparisons per rule type (mirrors C++ logic)
VALID_COMPARISONS = {
    RuleType.numVal: {Comparison.Eq, Comparison.Neq, Comparison.Lt, Comparison.Gt,
                      Comparison.LtEq, Comparison.GtEq},
    RuleType.txtVal: {Comparison.Eq, Comparison.Neq},
    RuleType.swVal: {Comparison.Eq, Comparison.Neq},
    RuleType.timeDiff: {Comparison.Eq, Comparison.Neq, Comparison.Lt, Comparison.Gt,
                        Comparison.LtEq, Comparison.GtEq},
    RuleType.elCompNum: {Comparison.Eq, Comparison.Neq, Comparison.Lt, Comparison.Gt,
                         Comparison.LtEq, Comparison.GtEq},
    RuleType.elCompTxt: {Comparison.Eq, Comparison.Neq},
    RuleType.elCompSw: {Comparison.Eq, Comparison.Neq},
    RuleType.ruleComp: {Comparison.Eq, Comparison.Neq, Comparison.And, Comparison.Nand,
                        Comparison.Or, Comparison.Nor, Comparison.Imply, Comparison.Nimply,
                        Comparison.Xor, Comparison.Xnor},
}


@dataclass
class Rule:
    """A single stateRuleEngine rule."""

    name: str
    rule_type: RuleType
    comparison: Comparison = Comparison.Eq
    priority: Priority = Priority.none
    message: str = ""

    # For single-property rules (numVal, txtVal, swVal, timeDiff)
    property: str = ""
    element: str = ""
    target: Optional[str] = None  # stored as string, written appropriately

    # For two-property rules (elCompNum, elCompTxt, elCompSw)
    property1: str = ""
    element1: str = ""
    property2: str = ""
    element2: str = ""

    # For numeric comparisons
    tol: Optional[float] = None

    # For ruleComp rules
    rule1: str = ""
    rule2: str = ""

    def validate(self):
        """Validate rule configuration. Raises ValueError on problems."""
        if not self.name:
            raise ValueError("Rule must have a name")

        if self.comparison not in VALID_COMPARISONS[self.rule_type]:
            raise ValueError(
                f"Comparison {self.comparison.value} is not valid for rule type "
                f"{self.rule_type.value} in rule '{self.name}'"
            )

        if self.rule_type in (RuleType.numVal, RuleType.txtVal, RuleType.swVal, RuleType.timeDiff):
            if not self.property:
                raise ValueError(f"Rule '{self.name}' ({self.rule_type.value}): property is required")
            if not self.element:
                raise ValueError(f"Rule '{self.name}' ({self.rule_type.value}): element is required")

        if self.rule_type in (RuleType.elCompNum, RuleType.elCompTxt, RuleType.elCompSw):
            for attr in ("property1", "element1", "property2", "element2"):
                if not getattr(self, attr):
                    raise ValueError(f"Rule '{self.name}' ({self.rule_type.value}): {attr} is required")

        if self.rule_type == RuleType.ruleComp:
            if not self.rule1 or not self.rule2:
                raise ValueError(f"Rule '{self.name}' (ruleComp): rule1 and rule2 are required")

        if self.rule_type == RuleType.swVal and self.target is not None:
            if self.target not in ("On", "Off"):
                raise ValueError(f"Rule '{self.name}' (swVal): target must be 'On' or 'Off', got '{self.target}'")


@dataclass
class StateRuleTree:
    """A collection of rules forming a state rule tree."""

    rules: list[Rule] = field(default_factory=list)

    def validate(self):
        """Validate all rules and cross-references."""
        names = {r.name for r in self.rules}

        # Check for duplicates
        if len(names) != len(self.rules):
            seen = set()
            for r in self.rules:
                if r.name in seen:
                    raise ValueError(f"Duplicate rule name: '{r.name}'")
                seen.add(r.name)

        for r in self.rules:
            r.validate()

            if r.rule_type == RuleType.ruleComp:
                if r.rule1 not in names:
                    raise ValueError(f"Rule '{r.name}': rule1 '{r.rule1}' not found")
                if r.rule2 not in names:
                    raise ValueError(f"Rule '{r.name}': rule2 '{r.rule2}' not found")
                if r.rule1 == r.name:
                    raise ValueError(f"Rule '{r.name}': rule1 cannot reference itself")
                if r.rule2 == r.name:
                    raise ValueError(f"Rule '{r.name}': rule2 cannot reference itself")

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def get_rule(self, name: str) -> Optional[Rule]:
        for r in self.rules:
            if r.name == name:
                return r
        return None
