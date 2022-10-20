#!/bin/bash
set -uo pipefail
sudo umount -f /dev/cpuset
# sudo mount -t cgroup -o cpuset none /dev/cpuset
podman build --security-opt=seccomp=unconfined -t xcontainer .
# podman run --rm --privileged -v /opt/MagAOX:/opt/MagAOX -v /data:/data -it xcontainer bash
# podman run --rm --privileged -v /data:/data -it xcontainer bash
podman run --rm --privileged -it xcontainer bash