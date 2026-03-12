"""Tests for the stateruletree package."""

import textwrap
from pathlib import Path

import pytest

from stateruletree.models import Comparison, Priority, Rule, StateRuleTree, RuleType
from stateruletree.parser import parse_drawio, _strip_html, _parse_label
from stateruletree.toml_writer import rules_to_conf_string

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_rule_validation_missing_property(self):
        r = Rule(name="test", rule_type=RuleType.swVal, property="", element="x")
        with pytest.raises(ValueError, match="property is required"):
            r.validate()

    def test_rule_validation_missing_element(self):
        r = Rule(name="test", rule_type=RuleType.swVal, property="dev.prop", element="")
        with pytest.raises(ValueError, match="element is required"):
            r.validate()

    def test_rule_validation_bad_comparison(self):
        r = Rule(name="test", rule_type=RuleType.swVal, property="dev.prop",
                 element="el", comparison=Comparison.Lt)
        with pytest.raises(ValueError, match="not valid for rule type"):
            r.validate()

    def test_rule_validation_swval_bad_target(self):
        r = Rule(name="test", rule_type=RuleType.swVal, property="dev.prop",
                 element="el", target="Maybe")
        with pytest.raises(ValueError, match="must be 'On' or 'Off'"):
            r.validate()

    def test_rule_validation_rulecomp_missing_rules(self):
        r = Rule(name="test", rule_type=RuleType.ruleComp, comparison=Comparison.And,
                 rule1="a", rule2="")
        with pytest.raises(ValueError, match="rule1 and rule2 are required"):
            r.validate()

    def test_rule_validation_two_prop_missing(self):
        r = Rule(name="test", rule_type=RuleType.elCompNum,
                 property1="a.b", element1="x", property2="", element2="y")
        with pytest.raises(ValueError, match="property2 is required"):
            r.validate()

    def test_tree_validates_cross_refs(self):
        tree = StateRuleTree(rules=[
            Rule(name="a", rule_type=RuleType.swVal, property="d.p", element="e", target="On"),
            Rule(name="b", rule_type=RuleType.ruleComp, comparison=Comparison.And,
                 rule1="a", rule2="nonexistent"),
        ])
        with pytest.raises(ValueError, match="rule2 'nonexistent' not found"):
            tree.validate()

    def test_tree_detects_self_reference(self):
        tree = StateRuleTree(rules=[
            Rule(name="a", rule_type=RuleType.swVal, property="d.p", element="e", target="On"),
            Rule(name="b", rule_type=RuleType.ruleComp, comparison=Comparison.And,
                 rule1="a", rule2="b"),
        ])
        with pytest.raises(ValueError, match="cannot reference itself"):
            tree.validate()

    def test_tree_detects_duplicates(self):
        tree = StateRuleTree(rules=[
            Rule(name="a", rule_type=RuleType.swVal, property="d.p", element="e", target="On"),
            Rule(name="a", rule_type=RuleType.txtVal, property="d.q", element="f", target="X"),
        ])
        with pytest.raises(ValueError, match="Duplicate rule name"):
            tree.validate()


# ---------------------------------------------------------------------------
# Parser helper tests
# ---------------------------------------------------------------------------

class TestParserHelpers:
    def test_strip_html_basic(self):
        assert _strip_html("<div><b>hello</b></div>") == "hello"

    def test_strip_html_br(self):
        assert _strip_html("a<br>b<br/>c<BR />d") == "a\nb\nc\nd"

    def test_strip_html_entities(self):
        assert _strip_html("a&amp;b&lt;c") == "a&b<c"

    def test_parse_label(self):
        label = "name=foo\nruleType=swVal\nproperty=d.p\nelement=e\ntarget=On"
        d = _parse_label(label)
        assert d["name"] == "foo"
        assert d["ruleType"] == "swVal"
        assert d["target"] == "On"

    def test_parse_label_with_html(self):
        label = "<div>name=foo<br>ruleType=swVal<br/>property=d.p</div>"
        d = _parse_label(label)
        assert d["name"] == "foo"
        assert d["ruleType"] == "swVal"

    def test_parse_label_ignores_freetext(self):
        label = "This is a comment\nruleType=numVal\nproperty=d.p"
        d = _parse_label(label)
        assert "ruleType" in d
        # "This is a comment" has no =, so ignored
        assert len(d) == 2


# ---------------------------------------------------------------------------
# Parser integration tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_parse_fwfpm_example(self):
        tree = parse_drawio(EXAMPLES_DIR / "fwfpm_stagesci1.drawio")

        assert len(tree.rules) == 5

        names = {r.name for r in tree.rules}
        assert "fwfpm-fpm" in names
        assert "fwfpm-READY" in names
        assert "fwfpm-fpm-READY" in names
        assert "fwfpm-stagesci1-neq" in names
        assert "fwfpm-fpm-stagesci-fpm" in names

        # Check leaf rules
        fpm = tree.get_rule("fwfpm-fpm")
        assert fpm.rule_type == RuleType.swVal
        assert fpm.property == "fwfpm.filterName"
        assert fpm.element == "fpm"
        assert fpm.target == "On"

        ready = tree.get_rule("fwfpm-READY")
        assert ready.rule_type == RuleType.txtVal
        assert ready.target == "READY"

        neq = tree.get_rule("fwfpm-stagesci1-neq")
        assert neq.rule_type == RuleType.elCompSw
        assert neq.comparison == Comparison.Neq
        assert neq.property1 == "fwfpm.filterName"
        assert neq.property2 == "stagesci1.presetName"

        # Check gates
        fpm_ready = tree.get_rule("fwfpm-fpm-READY")
        assert fpm_ready.rule_type == RuleType.ruleComp
        assert fpm_ready.comparison == Comparison.And
        assert {fpm_ready.rule1, fpm_ready.rule2} == {"fwfpm-fpm", "fwfpm-READY"}

        top = tree.get_rule("fwfpm-fpm-stagesci-fpm")
        assert top.priority == Priority.caution
        assert "stagesci1" in top.message
        assert {top.rule1, top.rule2} == {"fwfpm-fpm-READY", "fwfpm-stagesci1-neq"}

    def test_parse_inline_xml(self):
        xml = textwrap.dedent("""\
        <mxGraphModel>
          <root>
            <mxCell id="0" />
            <mxCell id="1" parent="0" />
            <mxCell id="r1" value="name=rule-a&#xa;ruleType=txtVal&#xa;property=dev.prop&#xa;element=state&#xa;target=OK" vertex="1" parent="1">
              <mxGeometry x="0" y="0" width="100" height="50" as="geometry" />
            </mxCell>
            <mxCell id="r2" value="name=rule-b&#xa;ruleType=txtVal&#xa;property=dev2.prop&#xa;element=state&#xa;target=OK" vertex="1" parent="1">
              <mxGeometry x="200" y="0" width="100" height="50" as="geometry" />
            </mxCell>
            <mxCell id="gate" value="name=combined&#xa;ruleType=ruleComp&#xa;comp=Or&#xa;priority=info&#xa;message=something is OK" vertex="1" parent="1">
              <mxGeometry x="100" y="100" width="100" height="50" as="geometry" />
            </mxCell>
            <mxCell id="e1" edge="1" source="r1" target="gate" parent="1" />
            <mxCell id="e2" edge="1" source="r2" target="gate" parent="1" />
          </root>
        </mxGraphModel>
        """)
        tree = parse_drawio(xml)
        assert len(tree.rules) == 3
        combined = tree.get_rule("combined")
        assert combined.comparison == Comparison.Or
        assert combined.priority == Priority.info

    def test_parse_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_drawio("/nonexistent/file.drawio")

    def test_parse_gate_wrong_children(self):
        """Gate with only 1 child should raise."""
        xml = textwrap.dedent("""\
        <mxGraphModel>
          <root>
            <mxCell id="0" />
            <mxCell id="1" parent="0" />
            <mxCell id="r1" value="name=rule-a&#xa;ruleType=txtVal&#xa;property=dev.prop&#xa;element=state&#xa;target=OK" vertex="1" parent="1" />
            <mxCell id="gate" value="name=my-gate&#xa;ruleType=ruleComp&#xa;comp=And" vertex="1" parent="1" />
            <mxCell id="e1" edge="1" source="r1" target="gate" parent="1" />
          </root>
        </mxGraphModel>
        """)
        with pytest.raises(ValueError, match="exactly 2 child rules"):
            parse_drawio(xml)

    def test_parse_missing_name(self):
        """Node without name= field should raise."""
        xml = textwrap.dedent("""\
        <mxGraphModel>
          <root>
            <mxCell id="0" />
            <mxCell id="1" parent="0" />
            <mxCell id="r1" value="ruleType=txtVal&#xa;property=dev.prop&#xa;element=state&#xa;target=OK" vertex="1" parent="1" />
          </root>
        </mxGraphModel>
        """)
        with pytest.raises(ValueError, match="missing required 'name' field"):
            parse_drawio(xml)


# ---------------------------------------------------------------------------
# TOML writer tests
# ---------------------------------------------------------------------------

class TestTomlWriter:
    def test_simple_swval(self):
        tree = StateRuleTree(rules=[
            Rule(name="test-sw", rule_type=RuleType.swVal,
                 property="dev.prop", element="el", target="On"),
        ])
        conf = rules_to_conf_string(tree)
        assert "[test-sw]" in conf
        assert "ruleType=swVal" in conf
        assert "property=dev.prop" in conf
        assert "element=el" in conf
        assert "target=On" in conf
        # Default comp (Eq) should not be written
        assert "comp=" not in conf

    def test_rulecomp_default_and_not_written(self):
        tree = StateRuleTree(rules=[
            Rule(name="a", rule_type=RuleType.swVal, property="d.p", element="e", target="On"),
            Rule(name="b", rule_type=RuleType.swVal, property="d.q", element="f", target="Off"),
            Rule(name="gate", rule_type=RuleType.ruleComp, comparison=Comparison.And,
                 rule1="a", rule2="b"),
        ])
        conf = rules_to_conf_string(tree)
        # And is default for ruleComp, should not appear
        assert "comp=" not in conf

    def test_rulecomp_or_written(self):
        tree = StateRuleTree(rules=[
            Rule(name="a", rule_type=RuleType.swVal, property="d.p", element="e", target="On"),
            Rule(name="b", rule_type=RuleType.swVal, property="d.q", element="f", target="Off"),
            Rule(name="gate", rule_type=RuleType.ruleComp, comparison=Comparison.Or,
                 rule1="a", rule2="b"),
        ])
        conf = rules_to_conf_string(tree)
        assert "comp=Or" in conf

    def test_topological_order(self):
        """ruleComp rules should appear after their children."""
        tree = StateRuleTree(rules=[
            Rule(name="top", rule_type=RuleType.ruleComp, comparison=Comparison.And,
                 rule1="mid", rule2="leaf-c"),
            Rule(name="mid", rule_type=RuleType.ruleComp, comparison=Comparison.Or,
                 rule1="leaf-a", rule2="leaf-b"),
            Rule(name="leaf-a", rule_type=RuleType.swVal, property="d.p", element="e", target="On"),
            Rule(name="leaf-b", rule_type=RuleType.swVal, property="d.q", element="f", target="Off"),
            Rule(name="leaf-c", rule_type=RuleType.txtVal, property="d.r", element="g", target="OK"),
        ])
        conf = rules_to_conf_string(tree)
        lines = conf.split("\n")
        sections = [l for l in lines if l.startswith("[")]
        # leaf-a and leaf-b before mid, mid and leaf-c before top
        idx = {s.strip("[]"): i for i, s in enumerate(sections)}
        assert idx["leaf-a"] < idx["mid"]
        assert idx["leaf-b"] < idx["mid"]
        assert idx["mid"] < idx["top"]
        assert idx["leaf-c"] < idx["top"]

    def test_numval_with_tol(self):
        tree = StateRuleTree(rules=[
            Rule(name="num-test", rule_type=RuleType.numVal,
                 property="dev.prop", element="val", target="3.14",
                 comparison=Comparison.GtEq, tol=0.001),
        ])
        conf = rules_to_conf_string(tree)
        assert "tol=0.001" in conf
        assert "comp=GtEq" in conf
        assert "target=3.14" in conf

    def test_elcomp_rule(self):
        tree = StateRuleTree(rules=[
            Rule(name="cmp", rule_type=RuleType.elCompSw,
                 property1="d1.p1", element1="e1",
                 property2="d2.p2", element2="e2",
                 comparison=Comparison.Neq),
        ])
        conf = rules_to_conf_string(tree)
        assert "property1=d1.p1" in conf
        assert "element1=e1" in conf
        assert "property2=d2.p2" in conf
        assert "element2=e2" in conf
        assert "comp=Neq" in conf

    def test_priority_and_message(self):
        tree = StateRuleTree(rules=[
            Rule(name="a", rule_type=RuleType.swVal, property="d.p", element="e", target="On"),
            Rule(name="b", rule_type=RuleType.swVal, property="d.q", element="f", target="Off"),
            Rule(name="alert-rule", rule_type=RuleType.ruleComp, comparison=Comparison.And,
                 rule1="a", rule2="b", priority=Priority.alert,
                 message="Something is very wrong"),
        ])
        conf = rules_to_conf_string(tree)
        assert "priority=alert" in conf
        assert "message=Something is very wrong" in conf

    def test_fwfpm_example_roundtrip(self):
        """Parse the draw.io example and check the output matches expected config."""
        tree = parse_drawio(EXAMPLES_DIR / "fwfpm_stagesci1.drawio")
        conf = rules_to_conf_string(tree)

        # Verify all expected sections exist
        assert "[fwfpm-fpm]" in conf
        assert "[fwfpm-READY]" in conf
        assert "[fwfpm-fpm-READY]" in conf
        assert "[fwfpm-stagesci1-neq]" in conf
        assert "[fwfpm-fpm-stagesci-fpm]" in conf

        # Verify key fields
        assert "ruleType=swVal" in conf
        assert "ruleType=txtVal" in conf
        assert "ruleType=elCompSw" in conf
        assert "ruleType=ruleComp" in conf
        assert "priority=caution" in conf
        assert "message=fwfpm is in fpm but stagesci1 is not in focus position fpm" in conf
