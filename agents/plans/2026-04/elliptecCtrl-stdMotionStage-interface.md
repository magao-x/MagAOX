The new app elliptecCtrl is being imported from a different fork of MagAO-X.  We need to make some updates to it:
- while it inherits from stdMotionStage, it does not use all of the interface and re-defines several parts of it (homing, stop, presets)
- it does not telemeter everything that it should.  some of this will get cleaned up by switching to being a full user of stdMotionStage interfaces.
- We also need to add tests.  AGENTS.md is now up to date on this branch with our latest policy developments.  You can also review the branches jrmales/tracker-crash-guards and jrmales/flowrpm-app for how we have brought the adcTracker and flowRPM apps up to standards.

Please review AGENTS.md, and then formulate a plan to address the above points.  Fill in the plan below, but do not modify this prompt.  Do not being executing until I have a chance to review the plan.

Plan:
