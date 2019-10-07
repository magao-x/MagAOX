#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail


function setgid_all() {
    # n.b. can't be recursive because g+s on files means something else
    # so we find all directories and individually chmod them:
    find $1 -type d -exec chmod g+s {} \;
}

function make_on_data_array() {
  # If run on instrument computer, make the name provided as an arg a link from $2/$1
  # to /data/$1.
  # If not on a real instrument computer, just make a normal folder under /opt/MagAOX/
  if [[ -z $1 ]]; then
    log_error "Missing target name argument for make_on_data_array"
    exit 1
  else
    TARGET_NAME=$1
  fi
  if [[ -z $2 ]]; then
    log_error "Missing parent dir argument for make_on_data_array"
    exit 1
  else
    PARENT_DIR=$2
  fi

  if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC ]]; then
    REAL_DIR=/data/$TARGET_NAME
    mkdir -pv $REAL_DIR
    link_if_necessary $REAL_DIR $PARENT_DIR/$TARGET_NAME
  else
    REAL_DIR=$PARENT_DIR/$TARGET_NAME
    mkdir -pv $REAL_DIR
  fi
  chown -RP xsup:magaox $REAL_DIR
  chmod -R u=rwX,g=rwX,o=rX $REAL_DIR
  setgid_all $REAL_DIR
}

mkdir -pv /opt/MagAOX
chown root:root /opt/MagAOX

mkdir -pv /opt/MagAOX/bin
# n.b. not using -R on *either* chown *or* chmod so we don't clobber setuid binaries
chown root:root /opt/MagAOX/bin
chmod u+rwX,g+rX,o+rX /opt/MagAOX/bin

if [[ "$MAGAOX_ROLE" == "vm" ]]; then
  mkdir -pv /vagrant/vm/calib
  link_if_necessary /vagrant/vm/calib /opt/MagAOX/calib
else
  mkdir -pv /opt/MagAOX/calib
  chown -R root:magaox-dev /opt/MagAOX/calib
  chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/calib
  setgid_all /opt/MagAOX/calib
fi

if [[ "$MAGAOX_ROLE" == "vm" ]]; then
  mkdir -pv /vagrant/vm/config
  link_if_necessary /vagrant/vm/config /opt/MagAOX/config
else
  mkdir -pv /opt/MagAOX/config
  chown -R root:magaox-dev /opt/MagAOX/config
  chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/config
  setgid_all /opt/MagAOX/config
fi

mkdir -pv /opt/MagAOX/drivers/fifos
chown -R root:root /opt/MagAOX/drivers
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/drivers
chown -R root:magaox /opt/MagAOX/drivers/fifos

make_on_data_array logs /opt/MagAOX
make_on_data_array rawimages /opt/MagAOX
make_on_data_array telem /opt/MagAOX

mkdir -pv /opt/MagAOX/secrets
chown -R root:root /opt/MagAOX/secrets
chmod -R u=rwX,g=,o= /opt/MagAOX/secrets

mkdir -pv /opt/MagAOX/source
chown -R root:magaox-dev /opt/MagAOX/source
# n.b. using + instead of = so we don't clobber setuid binaries
chmod -R u+rwX,g+rwX,o+rX /opt/MagAOX/source
setgid_all /opt/MagAOX/source

mkdir -pv /opt/MagAOX/sys
chown -R root:root /opt/MagAOX/sys
chmod -R u=rwX,g=rX,o=rX /opt/MagAOX/sys

mkdir -pv /opt/MagAOX/vendor
chown root:magaox-dev /opt/MagAOX/vendor
chmod u=rwX,g=rwX,o=rX /opt/MagAOX/vendor
setgid_all /opt/MagAOX/vendor

if [[ "$MAGAOX_ROLE" == "vm" ]]; then
  mkdir -pv /vagrant/vm/cache
  link_if_necessary /vagrant/vm/cache /opt/MagAOX/.cache
else
  mkdir -pv /opt/MagAOX/.cache
  chown -R root:magaox-dev /opt/MagAOX/.cache
  chmod -R u=rwX,g=rwsX,o=rX /opt/MagAOX/.cache
fi
