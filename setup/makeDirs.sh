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
Usage: makeDirs.sh [--dev]
Set up the MagAO-X folder structure, users, groups, and permissions.

  --dev   Set up for local development (i.e. don't assume real
          MagAO-X mount locations are present)
HERE
  exit 1
fi

mkdir  -pv /opt/MagAOX
mkdir  -pv /opt/MagAOX/bin
mkdir  -pv /opt/MagAOX/drivers
mkdir  -pv /opt/MagAOX/drivers/fifos
mkdir  -pv /opt/MagAOX/config

mkdir -pv "$LOGDIR"
if [[ ! $(getent group magaox) ]]; then
  groupadd magaox
  echo "Added group magaox"
else
  echo "Group magaox exists"
fi
chown :magaox "$LOGDIR"
chmod g+rw -v "$LOGDIR"
chmod g+s -v "$LOGDIR"

if [ "$LOGDIR" != "/opt/MagAOX/logs" ] ; then
  echo "Creating logs symlink . . ."
  ln -s "$LOGDIR" /opt/MagAOX/logs
fi

mkdir  -pv /opt/MagAOX/sys
mkdir  -pv /opt/MagAOX/secrets
chmod o-rwx -v /opt/MagAOX/secrets
chmod g-rwx -v /opt/MagAOX/secrets
