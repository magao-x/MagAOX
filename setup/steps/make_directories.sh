#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
IFS=$'\n\t'
TARGET_ENV=$1

function link_if_necessary() {
  thedir=$1
  thelinkname=$2
  if [[ "$thedir" != "$thelinkname" ]]; then
    if [[ -L $thelinkname ]]; then
      if [[ "$(readlink -- "$thelinkname")" != $thedir ]]; then
        echo "$thelinkname is an existing link, but doesn't point to $thedir. Aborting."
        exit 1
      fi
    elif [[ -e $thelinkname ]]; then
      echo "$thelinkname exists, but is not a symlink and we want logs in $thedir. Aborting."
      exit 1
    else
        ln -sv "$thedir" "$thelinkname"
    fi
  fi
}

mkdir -pv /opt/MagAOX
mkdir -pv /opt/MagAOX/bin
mkdir -pv /opt/MagAOX/drivers/fifos
mkdir -pv /opt/MagAOX/source
mkdir -pv /opt/MagAOX/vendor
mkdir -pv /opt/MagAOX/sys
mkdir -pv /opt/MagAOX/secrets

if [[ "$TARGET_ENV" == "vagrant" ]]; then
  mkdir -pv /opt/MagAOX/logs
  mkdir -pv /opt/MagAOX/rawimages
elif [[ "$TARGET_ENV" == "instrument" ]]; then
  mkdir -pv /data/logs
  link_if_necessary /data/logs /opt/MagAOX/logs
  mkdir -pv /data/rawimages
  link_if_necessary /data/rawimages /opt/MagAOX/rawimages
else
  echo "Unknown TARGET_ENV passed as argument 1"
  exit 1
fi
