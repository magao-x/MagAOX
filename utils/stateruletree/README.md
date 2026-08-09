# stateruletree — draw.io Diagram to stateRuleEngine Config Converter

`stateruletree` parses rule tree diagrams created in [draw.io](https://app.diagrams.net/) and converts them to XWCTk `stateRuleEngine`-compatible `.conf` files.

## Quick Start

```bash
# Convert a draw.io diagram to a .conf file
python -m stateruletree my_rules.drawio -o my_rules.conf

# Print to stdout
python -m stateruletree my_rules.drawio

# Validate only (no output)
python -m stateruletree --validate my_rules.drawio
```

## Diagram Conventions

The diagram is a **directed graph** where edges flow from child rules up to parent (gate) rules. Nodes use a **structured label format**: key=value pairs, one per line.  In draw.io, use `Shift+Enter` or `<br>` for line breaks within a label.

Every node **must** contain both `name=...` and `ruleType=...`.

### 1. Leaf Nodes (Value Comparisons)

These are the basic rules that compare INDI property elements against target values or against each other.

#### Single-property rules

| ruleType   | Description                          | Required fields                      | Optional            |
|:----------:|--------------------------------------|--------------------------------------|---------------------|
| `numVal`   | Compare number element to target     | `property`, `element`, `target`      | `comp`, `tol`       |
| `txtVal`   | Compare text element to target       | `property`, `element`, `target`      | `comp`              |
| `swVal`    | Compare switch state to On/Off       | `property`, `element`                | `target`, `comp`    |
| `timeDiff` | Compare time difference to target    | `property`, `element`, `target`      | `comp`, `tol`       |

Example label for a `swVal` leaf:

```
name=fwfpm-fpm
ruleType=swVal
property=fwfpm.filterName
element=fpm
target=On
```

#### Two-property rules

| ruleType     | Description                        | Required fields                                        | Optional      |
|:------------:|------------------------------------|--------------------------------------------------------|---------------|
| `elCompNum`  | Compare two number elements        | `property1`, `element1`, `property2`, `element2`       | `comp`, `tol` |
| `elCompTxt`  | Compare two text elements          | `property1`, `element1`, `property2`, `element2`       | `comp`        |
| `elCompSw`   | Compare two switch elements        | `property1`, `element1`, `property2`, `element2`       | `comp`        |

Example:

```
name=fwfpm-stagesci1-neq
ruleType=elCompSw
property1=fwfpm.filterName
element1=fpm
property2=stagesci1.presetName
element2=fpm
comp=Neq
```

### 2. Gate Nodes (Rule Composition)

Gate nodes combine exactly **two child rules** with a logical operator.  Like all nodes, they use the structured label format with `name=` and `ruleType=ruleComp`.

Example:

```
name=fwfpm-fpm-stagesci-fpm
ruleType=ruleComp
comp=And
priority=caution
message=fwfpm is in fpm but stagesci1 is not in focus position fpm
```

Valid `comp` values for gates:
`And`, `Nand`, `Or`, `Nor`, `Eq`/`Xnor`, `Neq`/`Xor`, `Imply`, `Nimply`

### Edges

Draw arrows **from each child rule to the gate** that combines them.
Each gate must have exactly **2 incoming edges**.

```
 [leaf-A]  -->  [AND gate]  -->  [top-level gate with priority]
 [leaf-B]  ------^  ^
                    |
 [leaf-C]  ---------|  (ERROR: 3 children!)
```

If you need to combine more than 2 rules, chain gates:

```
 [leaf-A]  -->  [AND-1]  -->  [AND-2]
 [leaf-B]  -------^             ^
 [leaf-C]  ---------------------|
```

### Node Naming

Every node **must** have a `name=...` field.  This name becomes the TOML section
heading (`[name]`) in the output `.conf` file and is used for cross-references
between `ruleComp` rules and their children.  Nodes without a `name` field will
cause a parse error.

### Priority & Messages

Only rules with `priority` set to something other than `none` will be published
by `stateRuleEngine`.  Valid priorities: `none`, `info`, `caution`, `warning`,
`alert`.

The `message` field provides the text shown to users when the rule triggers.

## Comparison Operators

| Operator | Meaning              | Valid for                            |
|:--------:|----------------------|--------------------------------------|
| `Eq`     | Equal                | all types                            |
| `Neq`    | Not equal            | all types                            |
| `Lt`     | Less than            | `numVal`, `timeDiff`, `elCompNum`    |
| `Gt`     | Greater than         | `numVal`, `timeDiff`, `elCompNum`    |
| `LtEq`   | Less than or equal   | `numVal`, `timeDiff`, `elCompNum`    |
| `GtEq`   | Greater than or equal| `numVal`, `timeDiff`, `elCompNum`    |
| `And`    | Logical AND          | `ruleComp`                           |
| `Nand`   | Logical NAND         | `ruleComp`                           |
| `Or`     | Logical OR           | `ruleComp`                           |
| `Nor`    | Logical NOR          | `ruleComp`                           |
| `Imply`  | Material implication | `ruleComp` (non-commutative)         |
| `Nimply` | Material nonimplication | `ruleComp` (non-commutative)      |
| `Xor`    | Exclusive OR         | `ruleComp`                           |
| `Xnor`   | Exclusive NOR        | `ruleComp`                           |

## Full Example

The example in `examples/fwfpm_stagesci1.drawio` implements the classic
fwfpm/stagesci1 focus-check scenario from the stateRuleEngine documentation.

Running:

```bash
python -m stateruletree examples/fwfpm_stagesci1.drawio
```

Produces:

```ini
[fwfpm-fpm]
ruleType=swVal
property=fwfpm.filterName
element=fpm
target=On

[fwfpm-READY]
ruleType=txtVal
property=fwfpm.fsm_state
element=state
target=READY

[fwfpm-fpm-READY]
ruleType=ruleComp
rule1=fwfpm-fpm
rule2=fwfpm-READY

[fwfpm-stagesci1-neq]
ruleType=elCompSw
comp=Neq
property1=fwfpm.filterName
element1=fpm
property2=stagesci1.presetName
element2=fpm

[fwfpm-fpm-stagesci-fpm]
ruleType=ruleComp
priority=caution
message=fwfpm is in fpm but stagesci1 is not in focus position fpm
rule1=fwfpm-fpm-READY
rule2=fwfpm-stagesci1-neq
```

## Python API

```python
from stateruletree import parse_drawio, write_conf, rules_to_conf_string

# Parse a diagram
tree = parse_drawio("my_rules.drawio")

# Write to file
write_conf(tree, "output.conf")

# Or get as string
conf_text = rules_to_conf_string(tree)
```
