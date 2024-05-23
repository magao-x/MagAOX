#!/bin/bash
if [[ "$EUID" != 0 ]]; then
    sudo -H bash $0 "$@"
    exit $?
fi
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh

echo "user.max_user_namespaces=10000" > /etc/sysctl.d/42-rootless.conf
sysctl --system