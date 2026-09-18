#!/usr/bin/env python3
"""
Command-line interface for stateruletree: convert draw.io diagrams to stateRuleEngine .conf files.

Usage::

    python -m stateruletree diagram.drawio -o rules.conf
    python -m stateruletree diagram.drawio  # prints to stdout

Validate only::

    python -m stateruletree --validate diagram.drawio
"""

import argparse
import sys
from pathlib import Path

from .parser import parse_drawio
from .toml_writer import rules_to_conf_string, write_conf


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="stateruletree",
        description=(
            "Convert a draw.io fault-tree / state-tree diagram into "
            "MagAO-X stateRuleEngine .conf files."
        ),
    )
    parser.add_argument(
        "input",
        help="Path to a .drawio or .xml diagram file",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output .conf file path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate the diagram, don't produce output.",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        state_rule_tree = parse_drawio(input_path)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    n_rules = len(state_rule_tree.rules)
    n_gates = sum(1 for r in state_rule_tree.rules if r.rule_type.value == "ruleComp")
    n_leaf = n_rules - n_gates

    if args.validate:
        print(
            f"Valid diagram: {n_rules} rules ({n_leaf} leaf, {n_gates} gates)",
            file=sys.stderr,
        )
        sys.exit(0)

    if args.output:
        write_conf(state_rule_tree, args.output)
        print(
            f"Wrote {n_rules} rules ({n_leaf} leaf, {n_gates} gates) to {args.output}",
            file=sys.stderr,
        )
    else:
        print(rules_to_conf_string(state_rule_tree))


if __name__ == "__main__":
    main()
