#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

FS_INOTIFY_LIMITS="fs.inotify.max_user_watches=524288"
if ! grep $FS_INOTIFY_LIMITS /etc/sysctl.conf; then
  echo $FS_INOTIFY_LIMITS | sudo tee -a /etc/sysctl.conf || exit 1
  sudo sysctl -p || exit 1
fi