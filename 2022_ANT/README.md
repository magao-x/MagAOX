# To-Do:  Brian T. Carcich; Ascending Node Technologies; starting ca. March, 2022

make it possible to start and stop drivers and associated INDI protocol communications on-the-fly

eliminate xindidriver intermediate pass-through processes

## Outline

- [] Goal:  independent INDI server and INDI driver processes
  - current dependency uses xindidriver as intermediary, forked by INDI server, to communicate directly with INDI device drivers
  - current model allows
    - starting and stopping independent INDI device drivers without affecting currently-running device drivers
    - stopping entire INDIserver/xindidriver process tree
    - restarting with a new group of xindidrivers via command line
- [] Step 1:  prototype INDI server that connects to **named** FIFOs,
  - instead of anonymous pipes that are STDIN/STDOUT/STDERR of forked xindidrivers
- [] Step 2:  decouple INDI server from xindidriver intermediaries
  - [] Step 2.1:  connect directly to named FIFOs of INDI device drivers (not xindidrivers)
    - no longer control (fork, clean up) any processes from main INDI server process,
      - because those xindidriver processes will be no more
    - INDI server needs only know the named FIFOs' names
  - [] Step 2.2:  re-write INDI server to have its own named FIFO to receive start and stop commands
    - this will make starting and stopping of INDI server communications independent of INDI device drivers
- [] Step 3:  Write INDI Resurrector to control INDI server and INDI device drivers
  - [] Step 3.1:  resurrector will pass named FIFO names to INDI server
  - [] Step 3.2:  Implement heart beat, so both process crash and process lock-up (race condition) can be fixed

## Branches
- feature/named-pipes-consolidate-indiserver (Step 1)

## Details
- Step 1
  - INDIS_NAMED_FIFO_DIR:
    - The name of the environment variable (envvar) holding the absolute path to indiSERVER<->XindiDRIVER **named** FIFOs.
    - Optional, no default
      - If envvar is not assigned, fall back to un-named inter-process pipes
    - As of 2022-05-23, this is only implemented in the **feature/named-pipes-consolidate-indiserver** branch
    - E.g. 

        export INDIS_NAMED_FIFO_DIR=/opt/MagAOX/drivers/fifos/named
