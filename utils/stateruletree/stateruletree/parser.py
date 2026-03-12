"""
Parser for .drawio / .xml diagrams into StateRuleTree models.

Draw.io Diagram Conventions
===========================

The diagram is a directed graph where edges flow from child rules up to parent
(composite) rules.  The parser determines edge direction from the source/target
attributes on each edge.

All nodes use the same **structured label format**: key=value pairs separated
by newlines (or ``<br>`` / ``<br/>`` in draw.io HTML labels).  Every node
**must** contain both ``name=...`` and ``ruleType=...``.

Node Types
----------

1. **Value-comparison nodes** (leaf nodes):
   These compare an INDI property element to a target value, or compare
   two INDI property elements to each other.

   Example labels::

       name=fwfpm-fpm
       ruleType=swVal
       property=fwfpm.filterName
       element=fpm
       target=On

       name=stage-position
       ruleType=numVal
       property=stageSci.position
       element=pos
       target=3.5
       tol=0.01
       comp=Eq

       name=fwfpm-stagesci1-neq
       ruleType=elCompSw
       property1=fwfpm.filterName
       element1=fpm
       property2=stagesci1.presetName
       element2=fpm
       comp=Neq

2. **Gate nodes** (composite / ruleComp nodes):
   These combine two child rules with a logical operator.
   Gate nodes use the same structured label format with
   ``ruleType=ruleComp``.

   Gate nodes must have exactly two incoming edges (from child rules).

   Example::

       name=focus-check
       ruleType=ruleComp
       comp=And
       priority=caution
       message=fwfpm is in fpm but stagesci1 is not in focus position fpm

Node Naming
-----------

Every node **must** have a ``name=...`` field in its label.  This name becomes
the TOML section heading (``[name]``) in the output ``.conf`` file and is used
for cross-references between ``ruleComp`` rules and their children.  Nodes
without a ``name`` field will cause a parse error.

Edge Direction
--------------

Edges go FROM child rules TO parent rules:
  child ---> parent (gate)

In draw.io, draw arrows from each input rule to the gate that combines them.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

from .models import (
    Comparison,
    Priority,
    Rule,
    StateRuleTree,
    RuleType,
)


def _strip_html(text: str) -> str:
    """Strip HTML tags from draw.io labels and normalize line breaks."""
    # draw.io wraps labels in <div>, <span>, <font>, etc.
    # Replace <br>, <br/>, <br /> with newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities
    text = html.unescape(text)
    return text.strip()


def _parse_label(label_raw: str) -> dict[str, str]:
    """Parse a structured label into a dict of key=value pairs.

    Lines without ``=`` are ignored (allows mixing free text).
    """
    label = _strip_html(label_raw)
    result = {}
    for line in label.split("\n"):
        line = line.strip()
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key:
                result[key] = value
    return result


def _build_rule_from_fields(fields: dict[str, str], cell_id: str) -> Rule:
    """Build a Rule from parsed key=value fields.

    Every node must have an explicit ``name`` field.
    """
    if "name" not in fields:
        raise ValueError(f"Node (cell {cell_id}): missing required 'name' field in label")

    name = fields["name"]
    rule_type_str = fields.get("ruleType", "")

    if not rule_type_str:
        raise ValueError(f"Node '{name}' (cell {cell_id}): missing ruleType in label")

    try:
        rule_type = RuleType(rule_type_str)
    except ValueError:
        raise ValueError(
            f"Node '{name}' (cell {cell_id}): unknown ruleType '{rule_type_str}'. "
            f"Valid types: {', '.join(t.value for t in RuleType)}"
        )

    comp_str = fields.get("comp", "Eq")
    # Normalize Xor/Xnor aliases
    comp_map = {v.value: v for v in Comparison}
    if comp_str not in comp_map:
        raise ValueError(
            f"Node '{name}' (cell {cell_id}): unknown comparison '{comp_str}'. "
            f"Valid: {', '.join(c.value for c in Comparison)}"
        )
    comparison = comp_map[comp_str]

    priority_str = fields.get("priority", "none")
    try:
        priority = Priority(priority_str)
    except ValueError:
        raise ValueError(
            f"Node '{name}' (cell {cell_id}): unknown priority '{priority_str}'. "
            f"Valid: {', '.join(p.value for p in Priority)}"
        )

    rule = Rule(
        name=name,
        rule_type=rule_type,
        comparison=comparison,
        priority=priority,
        message=fields.get("message", ""),
    )

    # Single-property fields
    rule.property = fields.get("property", "")
    rule.element = fields.get("element", "")
    rule.target = fields.get("target", None)

    # Two-property fields
    rule.property1 = fields.get("property1", "")
    rule.element1 = fields.get("element1", "")
    rule.property2 = fields.get("property2", "")
    rule.element2 = fields.get("element2", "")

    # Tolerance
    if "tol" in fields:
        try:
            rule.tol = float(fields["tol"])
        except ValueError:
            raise ValueError(f"Node '{name}' (cell {cell_id}): invalid tol '{fields['tol']}'")

    # ruleComp sub-rules (may be set from edges later)
    rule.rule1 = fields.get("rule1", "")
    rule.rule2 = fields.get("rule2", "")

    return rule


def parse_drawio(source: Union[str, Path]) -> StateRuleTree:
    """Parse a draw.io XML file into a StateRuleTree.

    Parameters
    ----------
    source : str or Path
        Path to a ``.drawio`` or ``.xml`` file, or an XML string.

    Returns
    -------
    StateRuleTree
        The parsed state rule tree with all rules populated and cross-referenced.

    Raises
    ------
    ValueError
        On parse errors, missing fields, or invalid graph structure.
    """
    source_str = str(source)

    # Determine if source is a file path or raw XML
    if source_str.lstrip().startswith("<"):
        xml_str = source_str
    else:
        path = Path(source_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        xml_str = path.read_text(encoding="utf-8")

    tree = ET.fromstring(xml_str)

    # draw.io files have structure: <mxfile><diagram><mxGraphModel><root>...
    # or just <mxGraphModel><root>...
    # Find the root element containing mxCell elements
    root_elem = tree.find(".//root")
    if root_elem is None:
        # Maybe the cells are directly under mxGraphModel
        root_elem = tree.find(".//mxGraphModel")
    if root_elem is None:
        # Try the element itself
        root_elem = tree

    cells = root_elem.findall(".//mxCell")
    if not cells:
        raise ValueError("No mxCell elements found in the diagram")

    # Separate nodes (vertices) and edges
    nodes: dict[str, ET.Element] = {}  # cell_id -> element
    edges: list[ET.Element] = []

    for cell in cells:
        cell_id = cell.get("id", "")
        # Skip the root cells (id 0 and 1 are draw.io internal)
        if cell_id in ("0", "1"):
            continue

        is_edge = cell.get("edge") == "1"
        source_id = cell.get("source", "")
        target_id = cell.get("target", "")

        if is_edge or (source_id and target_id):
            edges.append(cell)
        elif cell.get("vertex") == "1" or cell.get("value") is not None:
            # It's a node if it's marked as vertex or has a value attribute
            if cell_id:
                nodes[cell_id] = cell

    # Also check for UserObject elements (draw.io sometimes wraps cells)
    for obj in root_elem.findall(".//UserObject"):
        obj_id = obj.get("id", "")
        if obj_id and obj_id not in ("0", "1"):
            cell_child = obj.find("mxCell")
            if cell_child is not None:
                is_edge = cell_child.get("edge") == "1"
                if is_edge:
                    # Copy source/target to obj for uniform processing
                    obj.set("source", cell_child.get("source", ""))
                    obj.set("target", cell_child.get("target", ""))
                    edges.append(obj)
                else:
                    nodes[obj_id] = obj

    # Build edge map: target_id -> [source_ids] (children flowing into parent)
    children_of: dict[str, list[str]] = {}
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src and tgt:
            children_of.setdefault(tgt, []).append(src)

    # Parse each node into a Rule
    rule_map: dict[str, Rule] = {}  # cell_id -> Rule
    name_to_id: dict[str, str] = {}  # rule_name -> cell_id (for duplicate detection)

    for cell_id, cell_elem in nodes.items():
        label = cell_elem.get("value", "") or cell_elem.get("label", "") or ""

        if not label.strip():
            continue  # skip empty label nodes (decorative)

        fields = _parse_label(label)
        if not fields:
            continue  # Decorative text node
        if "ruleType" not in fields:
            # Could be a comment — skip
            continue
        rule = _build_rule_from_fields(fields, cell_id)

        if rule.name in name_to_id:
            raise ValueError(
                f"Duplicate rule name '{rule.name}' in cells "
                f"'{name_to_id[rule.name]}' and '{cell_id}'"
            )

        rule_map[cell_id] = rule
        name_to_id[rule.name] = cell_id

    # Wire up ruleComp rules from edges
    for cell_id, rule in rule_map.items():
        if rule.rule_type == RuleType.ruleComp:
            # If rule1/rule2 already set from label, keep them
            if rule.rule1 and rule.rule2:
                continue

            child_ids = children_of.get(cell_id, [])
            # Filter to only children that are actual rules
            child_ids = [cid for cid in child_ids if cid in rule_map]

            if len(child_ids) < 2:
                # Maybe edges go the other direction? Check if this node is a source
                # pointing to two targets
                reverse_children = []
                for edge in edges:
                    if edge.get("source") == cell_id:
                        tgt = edge.get("target", "")
                        if tgt in rule_map:
                            reverse_children.append(tgt)
                if len(reverse_children) >= 2:
                    child_ids = reverse_children

            if len(child_ids) != 2:
                raise ValueError(
                    f"Gate rule '{rule.name}' (cell {cell_id}) must have exactly "
                    f"2 child rules connected by edges, found {len(child_ids)}: "
                    f"{[rule_map[c].name for c in child_ids if c in rule_map]}"
                )

            rule.rule1 = rule_map[child_ids[0]].name
            rule.rule2 = rule_map[child_ids[1]].name

    # Build the StateRuleTree
    state_rule_tree = StateRuleTree()
    for rule in rule_map.values():
        state_rule_tree.add_rule(rule)

    state_rule_tree.validate()
    return state_rule_tree
