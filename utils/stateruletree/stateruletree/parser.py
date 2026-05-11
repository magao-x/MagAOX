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


def _load_xml_source(source: Union[str, Path]) -> str:
    """Return XML string from a file path or a raw XML string."""
    source_str = str(source)
    if source_str.lstrip().startswith("<"):
        return source_str
    path = Path(source_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def _find_root_element(tree: ET.Element) -> ET.Element:
    """Locate the draw.io <root> element that contains mxCell children."""
    root_elem = tree.find(".//root")
    if root_elem is None:
        root_elem = tree.find(".//mxGraphModel")
    if root_elem is None:
        root_elem = tree
    return root_elem


def _classify_cells(
    cells: list[ET.Element],
) -> tuple[dict[str, ET.Element], list[ET.Element]]:
    """Separate mxCell elements into nodes (vertices) and edges.

    Draw.io internal cells (id 0 and 1) are skipped.
    """
    nodes: dict[str, ET.Element] = {}
    edges: list[ET.Element] = []
    for cell in cells:
        cell_id = cell.get("id", "")
        if cell_id in ("0", "1"):
            continue
        is_edge = cell.get("edge") == "1"
        source_id = cell.get("source", "")
        target_id = cell.get("target", "")
        if is_edge or (source_id and target_id):
            edges.append(cell)
        elif cell.get("vertex") == "1" or cell.get("value") is not None:
            if cell_id:
                nodes[cell_id] = cell
    return nodes, edges


def _collect_user_objects(
    root_elem: ET.Element,
    nodes: dict[str, ET.Element],
    edges: list[ET.Element],
) -> None:
    """Add UserObject-wrapped cells into *nodes* or *edges* in-place."""
    for obj in root_elem.findall(".//UserObject"):
        obj_id = obj.get("id", "")
        if not obj_id or obj_id in ("0", "1"):
            continue
        cell_child = obj.find("mxCell")
        if cell_child is None:
            continue
        if cell_child.get("edge") == "1":
            obj.set("source", cell_child.get("source", ""))
            obj.set("target", cell_child.get("target", ""))
            edges.append(obj)
        else:
            nodes[obj_id] = obj


def _build_children_map(edges: list[ET.Element]) -> dict[str, list[str]]:
    """Build a mapping from parent cell-id to list of child cell-ids."""
    children_of: dict[str, list[str]] = {}
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src and tgt:
            children_of.setdefault(tgt, []).append(src)
    return children_of


def _parse_nodes_to_rules(
    nodes: dict[str, ET.Element],
) -> tuple[dict[str, Rule], dict[str, str]]:
    """Parse each diagram node into a Rule; detect duplicates."""
    rule_map: dict[str, Rule] = {}
    name_to_id: dict[str, str] = {}
    for cell_id, cell_elem in nodes.items():
        label = cell_elem.get("value", "") or cell_elem.get("label", "") or ""
        if not label.strip():
            continue
        fields = _parse_label(label)
        if not fields or "ruleType" not in fields:
            continue
        rule = _build_rule_from_fields(fields, cell_id)
        if rule.name in name_to_id:
            raise ValueError(
                f"Duplicate rule name '{rule.name}' in cells "
                f"'{name_to_id[rule.name]}' and '{cell_id}'"
            )
        rule_map[cell_id] = rule
        name_to_id[rule.name] = cell_id
    return rule_map, name_to_id


def _resolve_rulecomp_children(
    cell_id: str,
    rule: Rule,
    rule_map: dict[str, Rule],
    children_of: dict[str, list[str]],
    edges: list[ET.Element],
) -> None:
    """Resolve and assign rule1/rule2 for a single ruleComp rule."""
    if rule.rule1 and rule.rule2:
        return  # already set from label

    child_ids = [cid for cid in children_of.get(cell_id, []) if cid in rule_map]

    if len(child_ids) < 2:
        reverse_children = [
            edge.get("target", "")
            for edge in edges
            if edge.get("source") == cell_id and edge.get("target", "") in rule_map
        ]
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


def _wire_rulecomp_rules(
    rule_map: dict[str, Rule],
    children_of: dict[str, list[str]],
    edges: list[ET.Element],
) -> None:
    """Wire rule1/rule2 for all ruleComp gates from edge connections."""
    for cell_id, rule in rule_map.items():
        if rule.rule_type == RuleType.ruleComp:
            _resolve_rulecomp_children(cell_id, rule, rule_map, children_of, edges)


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
    xml_str = _load_xml_source(source)
    tree = ET.fromstring(xml_str)
    root_elem = _find_root_element(tree)

    cells = root_elem.findall(".//mxCell")
    if not cells:
        raise ValueError("No mxCell elements found in the diagram")

    nodes, edges = _classify_cells(cells)
    _collect_user_objects(root_elem, nodes, edges)
    children_of = _build_children_map(edges)
    rule_map, _ = _parse_nodes_to_rules(nodes)
    _wire_rulecomp_rules(rule_map, children_of, edges)

    state_rule_tree = StateRuleTree()
    for rule in rule_map.values():
        state_rule_tree.add_rule(rule)

    state_rule_tree.validate()
    return state_rule_tree
