#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

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

  chown -RP $instrument_user:$instrument_group $REAL_DIR
  chmod -R u=rwX,g=rwX,o=rX $REAL_DIR
  setgid_all $REAL_DIR
}

mkdir -pv /opt/MagAOX
mkdir -pv /opt/MagAOX/bin
mkdir -pv /opt/MagAOX/drivers/fifos
mkdir -pv /opt/MagAOX/secrets
mkdir -pv /opt/MagAOX/sys
mkdir -pv /opt/MagAOX/vendor
mkdir -pv /opt/MagAOX/source

if [[ "$MAGAOX_ROLE" == "vm" && "$VM_WINDOWS_HOST" == 0 ]]; then
  mkdir -pv "$VM_SHARED_FOLDER/calib"
  link_if_necessary "$VM_SHARED_FOLDER/calib" /opt/MagAOX/calib
else
  mkdir -pv /opt/MagAOX/calib
  chown -R root:$instrument_group /opt/MagAOX/calib
  chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/calib
  setgid_all /opt/MagAOX/calib
fi

if [[ "$MAGAOX_ROLE" == "vm" && "$VM_WINDOWS_HOST" == 0 ]]; then
  mkdir -pv "$VM_SHARED_FOLDER/config"
  link_if_necessary "$VM_SHARED_FOLDER/config" /opt/MagAOX/config
else
  mkdir -pv /opt/MagAOX/config
  chown -R root:$instrument_dev_group /opt/MagAOX/config
  chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/config
  setgid_all /opt/MagAOX/config
fi

if [[ "$MAGAOX_ROLE" == "vm" ]]; then
  mkdir -pv "$VM_SHARED_FOLDER/cache"
  link_if_necessary "$VM_SHARED_FOLDER/cache" /opt/MagAOX/.cache
else
  mkdir -pv /opt/MagAOX/.cache
  chown -R root:$instrument_dev_group /opt/MagAOX/.cache
  chmod -R u=rwX,g=rwsX,o=rX /opt/MagAOX/.cache
fi

if [[ $MAGAOX_ROLE != "vm" ]]; then

  chown root:root /opt/MagAOX
  # n.b. not using -R on *either* chown *or* chmod so we don't clobber setuid binaries
  chown root:root /opt/MagAOX/bin
  chmod u+rwX,g+rX,o+rX /opt/MagAOX/bin

  chown -R root:root /opt/MagAOX/drivers
  chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/drivers
  chown -R root:$instrument_group /opt/MagAOX/drivers/fifos

  make_on_data_array logs /opt/MagAOX
  make_on_data_array rawimages /opt/MagAOX
  make_on_data_array telem /opt/MagAOX
  make_on_data_array cacao /opt/MagAOX


  chown -R root:root /opt/MagAOX/secrets
  chmod -R u=rwX,g=,o= /opt/MagAOX/secrets


  chown -R root:$instrument_dev_group /opt/MagAOX/source
  # n.b. using + instead of = so we don't clobber setuid binaries
  chmod -R u+rwX,g+rwX,o+rX /opt/MagAOX/source
  setgid_all /opt/MagAOX/source


  chown -R root:root /opt/MagAOX/sys
  chmod -R u=rwX,g=rX,o=rX /opt/MagAOX/sys

  chown root:$instrument_dev_group /opt/MagAOX/vendor
  chmod u=rwX,g=rwX,o=rX /opt/MagAOX/vendor
  setgid_all /opt/MagAOX/vendor
fi

if [[ "$MAGAOX_ROLE" == "AOC" ]]; then
  make_on_data_array rtc /opt/MagAOX
  make_on_data_array icc /opt/MagAOX
fi
