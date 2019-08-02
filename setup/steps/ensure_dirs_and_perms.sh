#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
TARGET_ENV=$1
if ! [[ $TARGET_ENV == vm || $TARGET_ENV == instrument ]]; then
  echo "Unknown TARGET_ENV passed as argument 1"
  exit 1
fi
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
      echo "$thelinkname exists, but is not a symlink and we want the destination to be $thedir. Aborting."
      exit 1
    else
        ln -sv "$thedir" "$thelinkname"
    fi
  fi
}

function setgid_all() {
    # n.b. can't be recursive because g+s on files means something else
    # so we find all directories and individually chmod them:
    find $1 -type d -exec chmod g+s {} \;
}

mkdir -pv /opt/MagAOX
chown root:root /opt/MagAOX

mkdir -pv /opt/MagAOX/bin
chown -R root:magaox-dev /opt/MagAOX/bin
# n.b. using + instead of = so we don't clobber setuid binaries
chmod -R u+rwX,g+rwX,o+rX /opt/MagAOX/bin
setgid_all /opt/MagAOX/bin

mkdir -pv /opt/MagAOX/calib
chown -R root:magaox-dev /opt/MagAOX/calib
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/calib
setgid_all /opt/MagAOX/calib

mkdir -pv /opt/MagAOX/config
chown -R root:magaox-dev /opt/MagAOX/config
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/config
setgid_all /opt/MagAOX/config

mkdir -pv /opt/MagAOX/drivers/fifos
chown -R root:root /opt/MagAOX/drivers
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/drivers

if [[ "$TARGET_ENV" == "vm" ]]; then
  REAL_LOGS_DIR=/opt/MagAOX/logs
  mkdir -pv $REAL_LOGS_DIR
elif [[ "$TARGET_ENV" == "instrument" ]]; then
  REAL_LOGS_DIR=/data/logs
  mkdir -pv $REAL_LOGS_DIR
  link_if_necessary $REAL_LOGS_DIR /opt/MagAOX/logs
fi
chown -RP xsup:magaox $REAL_LOGS_DIR
chmod -R u=rwX,g=rwX,o=rX $REAL_LOGS_DIR
setgid_all $REAL_LOGS_DIR

if [[ "$TARGET_ENV" == "vm" ]]; then
  REAL_RAWIMAGES_DIR=/opt/MagAOX/rawimages
  mkdir -pv $REAL_RAWIMAGES_DIR
elif [[ "$TARGET_ENV" == "instrument" ]]; then
  REAL_RAWIMAGES_DIR=/data/rawimages
  mkdir -pv $REAL_RAWIMAGES_DIR
  link_if_necessary $REAL_RAWIMAGES_DIR /opt/MagAOX/rawimages
fi
chown -RP xsup:magaox $REAL_RAWIMAGES_DIR
chmod -R u=rwX,g=rwX,o=rX $REAL_RAWIMAGES_DIR
setgid_all $REAL_RAWIMAGES_DIR

mkdir -pv /opt/MagAOX/secrets
chown -R root:root /opt/MagAOX/secrets
chmod -R u=rwX,g=,o= /opt/MagAOX/secrets

mkdir -pv /opt/MagAOX/source
chown -R root:magaox-dev /opt/MagAOX/source
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/source
setgid_all /opt/MagAOX/source

mkdir -pv /opt/MagAOX/sys
chown -R root:root /opt/MagAOX/sys
chmod -R u=rwX,g=rX,o=rX /opt/MagAOX/sys

mkdir -pv /opt/MagAOX/vendor
chown -R root:magaox-dev /opt/MagAOX/vendor
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/vendor
setgid_all /opt/MagAOX/vendor
