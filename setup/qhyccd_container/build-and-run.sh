#!/bin/bash
set -uo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
if [[ ! -e sdk_linux64_22.10.14.tgz ]]; then
    log_error "Download sdk_linux64_22.10.14.tgz first"
fi
podman build --security-opt=seccomp=unconfined -t qhyccd . || exit_with_error "Podman build failed"
podman run --rm --privileged -it qhyccd bash