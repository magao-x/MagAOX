#!/bin/bash
set -uo pipefail
podman build --security-opt=seccomp=unconfined -t xcontainer .
podman run --rm --privileged -v /opt/MagAOX:/opt/MagAOX -v /data:/data -it xcontainer bash