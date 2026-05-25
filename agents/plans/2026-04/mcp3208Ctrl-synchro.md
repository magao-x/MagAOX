The mcp3208Ctrl reads an A/D converter at a specified frequency.  It uses a simple integrator control loop to tune the timings of its internal loop to match the desired frequency.  See
int mcp3208Ctrl::acquireAndCheckValid.  Now we want to add a mode where the controller instead responds to an ImageStreamIO image stream semaphore to synchronize.  So it should wait on a semaphore, and read the A/D when the semaphore is incremented.  We also want an optional delay after the read, specified in microseconds.

Review AGENTS.md, and then analyze the problem and formulate a plan.  Please describe the plan below, and do not alter this prompt.  Do not begin implementing until after I have reviewed the plan.

Plan:
- 2026-04-10 execution note: `synchro.postDelay` is intended to delay the accelerometer read after the synchronization semaphore, not to delay publish after the read. The synchronized path should timestamp and read after that pre-read offset, and should use a simple feedback loop to trim the live sleep toward the requested semaphore-to-read delay.
