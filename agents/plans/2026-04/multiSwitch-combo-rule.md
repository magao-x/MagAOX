# Extending stateRuleEngine with multiSwitchCombo rule

## Problem to solve 

Our science camera focus stages have `presets` named like `65-35-lyotlg-ri`, which corresponds to positions of components as documented in INDI properties.  In this case it is the switches `stagebs.presetName.65-35=On`, `fwfpm.filterName.lyotlg=On`, and `stagescibs.presetName.ri=On`.  We need to add a `rule` in `stateRuleEngine` that takes <N> such multi-switch properties and combines the elements names to make a target (in this case the target would be `65-35-lyotlg-ri`), and compare that target to the preset position of designated stage.  In this case we want `stagesci1.presetName.65-35-lyotlg-ri=On`.  The TOML configuration should match the existing style as closely as possible.  We should specify the target format with std::format like syntax: "{}-{}-{}" in this case.  So the shape is something like: specify the number of switches to combine, the switch device.property specification, the format, and the target switch device.property.  The rule is on if the formatted combo does not match the target switch position. We can call this the `multiSwitchCombo` rule type.  Please formulate a plan and document below under the Plan section.  Do not modify this paragraph.  Do not begin implementing until I have approved the plan. Review AGENTS.md before starting.

## Plan

# `stateRuleEngine` `multiSwitchCombo` Rule Plan

## Summary
- Add a new `multiSwitchCombo` rule type to `stateRuleEngine` that reads the active element name from `N` source switch properties, formats them into one combo string, and compares that derived string to the active element name of a target switch property.
- `zero On` in any source or target switch property is not an error; it resolves to an empty string for comparison.
- `multiple On` in any source or target switch property also resolves to an empty string, but should emit a `software_error` log once when that property enters the multi-On state, then stay quiet until it recovers and breaks again.
- Expected touched files: `apps/stateRuleEngine/indiCompRules.hpp`, `apps/stateRuleEngine/indiCompRuleConfig.hpp`, `apps/stateRuleEngine/stateRuleEngine.hpp`, `apps/stateRuleEngine/stateRuleEngine.md`, `apps/stateRuleEngine/tests/indiCompRules_test.cpp`, `apps/stateRuleEngine/tests/indiCompRuleConfig_test.cpp`, and this plan file under `## Plan` if the approved plan is copied in.

## Public / Config Interface
- Introduce `ruleType=multiSwitchCombo`.
- Required keys:
  - `numSwitches`
  - `property1` through `propertyN`
  - `format`
  - `targetProperty`
- Optional keys:
  - `priority`
  - `message`
  - `comp`
- Supported comparisons: `Eq` and `Neq` only.
- Default `comp` to `Neq` for this rule type so the rule naturally goes `true` on mismatch.
- Example:
```toml
[stagesci1-focus-mismatch]
ruleType=multiSwitchCombo
priority=caution
message=stagesci1 preset does not match the current science-camera combo
numSwitches=3
property1=stagebs.presetName
property2=fwfpm.filterName
property3=stagescibs.presetName
format="{}-{}-{}"
targetProperty=stagesci1.presetName
```

## Implementation Changes
- Add a new `multiSwitchComboRule` in `indiCompRules.hpp` that stores:
  - the source switch-property pointers in order
  - the format string
  - the target switch-property pointer
  - per-property multi-On latch state so repeated invalid loops do not spam logs
- Add a small helper that resolves the “active name” of a switch property with these rules:
  - no elements `On` -> return `""`, no log
  - exactly one element `On` -> return that element name
  - more than one element `On` -> return `""` and mark a pending runtime diagnostic on first entry into that state
- Apply the same active-name resolution rules to the target property as to the sources.
- Keep `valid()` for `multiSwitchComboRule` focused on static configuration checks:
  - nonzero `numSwitches`
  - all source and target pointers are present and are switch properties
  - `format` contains exactly `numSwitches` plain `{}` placeholders
  - `comp` is `Eq` or `Neq`
- Implement combo formatting with literal `{}` substitution only; do not support the full `std::format` mini-language in v1.
- Add a non-throwing runtime-diagnostic hook to `indiCompRule` with a default no-op implementation.
  - `multiSwitchComboRule` uses it to expose a pending multi-On warning message.
  - `stateRuleEngine::appLogic()` drains and logs that warning after evaluating the rule, without converting the rule result into an exception path.
- Extend `loadRuleConfig(...)` in `indiCompRuleConfig.hpp` with a `multiSwitchCombo` branch that:
  - reads `numSwitches`
  - loops over `property1...propertyN`
  - reads `targetProperty`
  - strips optional surrounding quotes/whitespace from `format`
  - applies the `Neq` default for `comp` on this rule type
- Update `stateRuleEngine.md` to document the new rule type, new keywords, supported comparisons, zero-/multi-On behavior, and a science-camera combo example.

## Test Plan
- Extend `indiCompRuleConfig_test.cpp` to cover:
  - default `multiSwitchCombo` parsing
  - explicit `comp=Eq`
  - invalid `numSwitches`
  - missing `propertyK` within the declared range
  - non-switch property type errors
  - invalid comparison operator
  - format placeholder-count mismatch
- Extend `indiCompRules_test.cpp` to cover:
  - normal match and mismatch cases
  - embedded hyphens in source element names
  - zero-On source resolving to an empty string without error
  - zero-On target resolving to an empty string without error
  - multi-On source resolving to an empty string while still producing a comparison result
  - multi-On target resolving to an empty string while still producing a comparison result
  - runtime warning hook behavior: one warning on entry, none repeated, clear on recovery, and warn again on a later re-entry
- Run `clang-format -i` on all touched `stateRuleEngine` files and execute the targeted `stateRuleEngine` rule/config unit tests.

## Assumptions
- v1 uses only plain `{}` placeholders and substitutes source names positionally.
- Empty-string substitution is the correct transition-safe behavior for both zero-On and multi-On switch vectors.
- The multi-On warning should be emitted on state change only, not every loop and not on zero-On transitions.
- Comparing the derived combo string against the target property’s currently active element name is the intended behavior; for normal preset vectors this is equivalent to testing whether `targetProperty.<derived-name>` is `On`.
