We need to add a 2nd ImageStreamIO stream as an output of ocam2KCtrl.  It will be an empty, or minimum size, stream.  The reason to do so is to have a 2nd meta-data, namely semaphore, only stream structure that can be used to synchronize operations across computers.  It should be initialized and prepared in parallel to the main dev::framegrabber setup, and the semaphores should be incremented immediately after the main stream semaphores are incremented.

As a 2nd pass to this effort, we need to add tests for ocam2KCtrl.  Currently only the ocamUtils parsers are tested.  

Please update this document with a plan, below.  Do not alter this prompt, and do not being executing until I have reviewed the plan.

Plan:
