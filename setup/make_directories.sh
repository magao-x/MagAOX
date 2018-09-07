#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

envswitch=${1:---prod}

if [[ "$envswitch" == "--dev" ]]; then
  LOGDIR="${LOGDIR:-/opt/MagAOX/logs}"
elif [[ "$envswitch" == "--prod" ]]; then
  LOGDIR="${LOGDIR:-/data/logs}"
else
  cat <<'HERE'
Usage: make_directories.sh [--dev]
Set up the MagAO-X folder structure, users, groups, and permissions.

  --prod  (default) Set up for production (i.e. logs are on
          /data/logs and symlinked to /opt/MagAOX/logs)
  --dev   Set up for local development (i.e. don't assume real
          MagAO-X mount locations are present)
HERE
  exit 1
fi

mkdir  -pv /opt/MagAOX
mkdir  -pv /opt/MagAOX/bin
mkdir  -pv /opt/MagAOX/drivers
mkdir  -pv /opt/MagAOX/drivers/fifos
mkdir  -pv /opt/MagAOX/source
mkdir  -pv /opt/MagAOX/source/dependencies
mkdir  -pv /opt/MagAOX/sys
mkdir  -pv /opt/MagAOX/secrets
mkdir  -pv "$LOGDIR"

log_target=/opt/MagAOX/logs
if [ "$LOGDIR" != "$log_target" ] ; then
  if [ -L $log_target ]; then
    if [ "$(readlink -- "$path")" != $log_target ]; then
      echo "$log_target is an existing link, but doesn't point to $LOGDIR. Aborting."
      exit 1
    fi
  elif [ -e $log_target ]; then
    echo "$log_target exists, but is not a symlink and we want logs in $LOGDIR. Aborting."
    exit 1
else
    ln -sv "$LOGDIR" "$log_target"
  fi
fi
