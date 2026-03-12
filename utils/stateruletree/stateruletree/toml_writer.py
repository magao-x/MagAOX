"""
TOML writer for stateRuleEngine rules.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Union

from .models import (
    Comparison,
    Priority,
    Rule,
    StateRuleTree,
    RuleType,
)


def _rule_to_conf_lines(rule: Rule) -> list[str]:
    """Convert a single Rule to lines of .conf text (without the section header)."""
    lines = []
    lines.append(f"ruleType={rule.rule_type.value}")

    if rule.priority != Priority.none:
        lines.append(f"priority={rule.priority.value}")

    if rule.message:
        lines.append(f"message={rule.message}")

    # Write comp if non-default
    # Default is Eq for most types, And for ruleComp
    default_comp = Comparison.And if rule.rule_type == RuleType.ruleComp else Comparison.Eq
    if rule.comparison != default_comp:
        lines.append(f"comp={rule.comparison.value}")

    if rule.rule_type in (RuleType.numVal, RuleType.txtVal, RuleType.swVal, RuleType.timeDiff):
        lines.append(f"property={rule.property}")
        lines.append(f"element={rule.element}")
        if rule.target is not None:
            lines.append(f"target={rule.target}")

    if rule.rule_type in (RuleType.elCompNum, RuleType.elCompTxt, RuleType.elCompSw):
        lines.append(f"property1={rule.property1}")
        lines.append(f"element1={rule.element1}")
        lines.append(f"property2={rule.property2}")
        lines.append(f"element2={rule.element2}")

    if rule.rule_type in (RuleType.numVal, RuleType.timeDiff, RuleType.elCompNum):
        if rule.tol is not None:
            lines.append(f"tol={rule.tol}")

    if rule.rule_type == RuleType.ruleComp:
        lines.append(f"rule1={rule.rule1}")
        lines.append(f"rule2={rule.rule2}")

    return lines


def _topological_sort(rules: list[Rule]) -> list[Rule]:
    """Sort rules so that ruleComp rules appear after all rules they reference.

    This ensures the .conf file can be read top to bottom with forward
    references only.
    """
    name_to_rule = {r.name: r for r in rules}
    visited: set[str] = set()
    result: list[Rule] = []

    def visit(name: str):
        if name in visited:
            return
        visited.add(name)
        rule = name_to_rule.get(name)
        if rule is None:
            return
        if rule.rule_type == RuleType.ruleComp:
            visit(rule.rule1)
            visit(rule.rule2)
        result.append(rule)

    for rule in rules:
        visit(rule.name)

    return result


def rules_to_conf_string(tree: StateRuleTree) -> str:
    """Convert a StateRuleTree to a .conf file string.

    Parameters
    ----------
    tree : StateRuleTree
        The state rule tree to serialize.

    Returns
    -------
    str
        The .conf file content.
    """
    tree.validate()

    sorted_rules = _topological_sort(tree.rules)

    buf = StringIO()
    for i, rule in enumerate(sorted_rules):
        if i > 0:
            buf.write("\n")
        buf.write(f"[{rule.name}]\n")
        for line in _rule_to_conf_lines(rule):
            buf.write(f"{line}\n")

    return buf.getvalue()


def write_conf(tree: StateRuleTree, output: Union[str, Path]):
    """Write a StateRuleTree to a .conf file.

    Parameters
    ----------
    tree : StateRuleTree
        The state rule tree to write.
    output : str or Path
        Output file path.
    """
    content = rules_to_conf_string(tree)
    Path(output).write_text(content, encoding="utf-8")
