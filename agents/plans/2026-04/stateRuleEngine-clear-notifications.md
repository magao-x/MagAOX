# Extending stateRuleEngine with clear notifications

## Problem to solve 

After testing `multiSwitchCombo` on the instrument, it is clear that it would be helpful for `stateRuleEngine` to report when a rule which had been `On` turns `Off`.  This should be a one-time notification on the `On` to `Off` transition, probably as an `INFO` message, with the format `Cleared: <message>`.  If the rule does not have an explicit `message`, the notification should fall back to the rule name in the same way the existing active notifications do.  Please formulate a plan and document below under the Plan section.  Do not modify this paragraph.  Do not begin implementing until I have approved the plan. Review AGENTS.md before starting.

## Plan

# `stateRuleEngine` Clear Notification Plan

## Summary
- Add a clear notification in `stateRuleEngine` when a published rule transitions from `On` to `Off`.
- Send the clear notification as an informational message with text `INFO: Cleared: <detail>`, where `<detail>` is the configured rule message when present and otherwise the rule name.
- Keep the change local to `stateRuleEngine` runtime reporting, the handbook documentation, the application unit test, and this plan record.

## Implementation Changes
- In `apps/stateRuleEngine/stateRuleEngine.hpp`, use the existing published rule state (`info`, `caution`, `warning`, or `alert` property) as the transition latch:
  - read whether the rule element was previously `On`
  - evaluate the rule
  - update the published switch state
  - if the new value is `false` and the previous published state was `On`, send one clear notification
- Factor the notification formatting into small helper functions in `stateRuleEngine.hpp` so the behavior is explicit and unit-testable:
  - choose the published property from the rule priority
  - detect whether a rule element is currently `On`
  - format active notifications as today
  - format clear notifications as `INFO: Cleared: <detail>`
- Keep the notification send path overridable so the application unit test can capture clear messages without requiring a live INDI driver.
- Preserve the current active-notification behavior for `val == true`, including use of the rule priority label and the `messageCount`/`timeToSend` logic.
- Preserve the current reset behavior after `val == false` by continuing to call `messageCount(0)` so future activations can notify again.
- Do not add any new rule configuration keys or rule-type changes for this feature.

## Test Plan
- Extend `apps/stateRuleEngine/tests/stateRuleEngine_test.cpp` with focused helper-level tests for:
  - explicit-message clear formatting
  - rule-name fallback clear formatting
  - active notification formatting still using the priority label
  - published-property selection for each rule priority
  - detection of whether a published rule element is currently `On`
- Add an `appLogic()` transition test that captures notifications and verifies one clear report for an observed `On -> Off` transition, with no repeat report once the published state is already `Off`.
- Keep the existing placeholder construction test.
- Run `clang-format -i` on touched files and execute the targeted `stateRuleEngine` unit test binary.

## Assumptions
- Clear notifications should apply to every published rule priority, but the clear message itself should always be sent as `INFO`.
- A clear notification should be emitted only on an observed `On -> Off` transition, not on repeated `Off` loops.
- Clear notifications should not be rate-limited independently; the transition itself is the one-shot gate.
- A rule that becomes true and false entirely between `appLogic()` loops will not generate a clear notification unless `stateRuleEngine` actually observed it in the `On` state first.
