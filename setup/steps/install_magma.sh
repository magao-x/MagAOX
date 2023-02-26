#!/bin/bash
# If not started as root, sudo yourself
if [[ "$EUID" != 0 ]]; then
    sudo bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
cd /opt/MagAOX/vendor
MAGMA_VERSION=2.5.4
MAGMA_FOLDER=./magma-$MAGMA_VERSION
if [[ ! -d $MAGMA_FOLDER ]]; then
  _cached_fetch http://icl.utk.edu/projectsfiles/magma/downloads/magma-$MAGMA_VERSION.tar.gz magma-$MAGMA_VERSION.tar.gz
  tar xzf magma-$MAGMA_VERSION.tar.gz
fi
cd $MAGMA_FOLDER
if [[ $(uname -i) == "x86_64" ]]; then
  cp -n make.inc-examples/make.inc.mkl-gcc ./make.inc
else
  cp -n make.inc-examples/make.inc.openblas ./make.inc
  sed -iE 's_#OPENBLASDIR ?= /usr/local/openblas_OPENBLASDIR ?= $(shell pkg-config --variable=libdir openblas)/../_g' ./make.inc
fi
# Limit target architecture to Pascal (1080Ti) and Volta (2080Ti) to save some compilation time
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC ]]; then
  if ! grep "GPU_TARGET = Pascal Volta" make.inc; then
    echo "GPU_TARGET = Pascal Volta" >> make.inc
  fi
elif [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == ci ]]; then
  if ! grep "GPU_TARGET = Pascal" make.inc; then
    echo "GPU_TARGET = Pascal" >> make.inc
  fi
fi
make -j 32
make install
echo "/usr/local/magma/lib" | sudo tee /etc/ld.so.conf.d/magma.conf
ldconfig
echo "# Configure MAGMA library environment variables (do not edit, see /opt/MagAOX/source/MagAOX/setup/install_magma.sh)" > /etc/profile.d/magma.sh
echo "export PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:/usr/local/magma/lib/pkgconfig" >> /etc/profile.d/magma.sh
