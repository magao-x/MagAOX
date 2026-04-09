We need to add a 2nd ImageStreamIO stream as an output of ocam2KCtrl.  It will be an empty, or minimum size, stream.  The reason to do so is to have a 2nd meta-data, namely semaphore, only stream structure that can be used to synchronize operations across computers.  It should be initialized, prepared, and maintained as part of the main dev::framegrabber setup, and the semaphores should be incremented immediately after the main stream semaphores are incremented.

As a 2nd pass to this effort, we need to add tests for ocam2KCtrl.  Currently only the ocamUtils parsers are tested.  Please follow the standards recently developed for flowRPM and adcTracker and elliptecCtrl (the last two are on feature branches).

Review AGENTS.md, then lease update this document with a plan, below.  Do not alter this prompt, and do not being executing until I have reviewed the plan.  Use the feature branch jrmales/ocam2k-sync.

Plan:
