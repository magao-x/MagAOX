# pupilCorAlign registration and readiness

The driver publishes its existing INDI properties after setup. Previously setup constructed an independent camera client and waited for camera properties, so a delayed server could prevent driver registration.

Use the app's existing camera subscription and defer the first science-stream attachment to the loop. The existing FSM reports `NOTCONNECTED` while waiting for INDI, `CONNECTED` while waiting for the science stream and `READY` when idle with the camera attached. A request received during startup is discarded. A server disconnect returns the mode selection to idle. DM availability and local calibrations follow the original app assumptions.

The regression test exercises a seven-second startup delay, discovery from another client, late camera attachment and a server restart using simulated transports. On an isolated Linux workstation, start the app before the configured server, record its PID, then restart the server after more than five seconds. Verify an independent client discovers the idle driver and that the original process reaches `READY` when the science stream is available.

The integration branch merges `xcam_shmim_fix` and `warm_cold_fix`. The latter supplies `dm.offset`, the matching probe patterns and the coronagraph alignment GUI changes. Its commit `5b74f1da` also supplies the measured shift relative to the actuator reference and the INDI repeat-count control. The regression test loads the deployed TOML before starting the driver and checks that `calibration.gain`, `calibration.nrepeat` and `calibration.nstack` reach the alignment loop. When omitted, these settings default to 0.25, one probe pair and four frames.
