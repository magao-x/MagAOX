We need to add a feature to cameraGUI that sends a command to a focus stage when a button is pressed.  We should add an optional feature to stdCamera called hasFocus.  stdCamera should then call a derived class function called "checkFocus" that returns true if in focus, false if not.  An INDI property switch "focus.state" should be exposed that is On when in focus, Off when not in focus. stdCamera should then also expose a "goto_focus.request" switch that when pressed calls a derived-class provided gotoFocus() function.

stdCamera should also provide some helper functions for the most common way that derived classes will work.
- Configure and set up to monitor a given INDI property switch, and if the switch is `On` it is interpreted as meaning the camera is out of focus.  The derived class can then return the status of this switch from within its checkFocus by calling checkFocusSwitchState().
- Configure and set up to use a multiSwitchCombo rule just like in stateRuleEngine, except now instead of comparison, the derived switch state is sent as an INDI command.  A stdCamera helper function is then called by the derived() class when gotoFocus() is called by stdCamera.

Finally, in cameraGUI a focus button should appear if a camera exposes the goto_focus.request switch.  The button should be disabled if focus.state is On (in-focus) and enabled if Off. This may require moving the shutter control down one when so-configured.

Review AGENTS.md.  Do not make changes to this prompt, but fill in the plan below in this document and commit it along with other work.

Plan:
