#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
MAGMA_VERSION=2.5.1-alpha1
MAGMA_FOLDER=./magma-$MAGMA_VERSION
if [[ ! -d $MAGMA_FOLDER ]]; then
  _cached_fetch http://icl.utk.edu/projectsfiles/magma/downloads/magma-$MAGMA_VERSION.tar.gz magma-$MAGMA_VERSION.tar.gz
  tar xzf magma-$MAGMA_VERSION.tar.gz
fi
cd $MAGMA_FOLDER
cp -n make.inc-examples/make.inc.mkl-gcc ./make.inc
# Limit target architecture to Pascal (1080Ti) and Volta (2080Ti) to save some compilation time
if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC ]]; then
  if ! grep "GPU_TARGET = Pascal Volta" make.inc; then
    echo "GPU_TARGET = Pascal Volta" >> make.inc
  fi
elif [[ $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == ci ]]; then
  if ! grep "GPU_TARGET = Pascal" make.inc; then
    echo "GPU_TARGET = Pascal" >> make.inc
  fi
fi
make -j 32
make install
echo "# Configure MAGMA library environment variables (do not edit, see /opt/MagAOX/source/MagAOX/setup/install_magma.sh)" > /etc/profile.d/magma.sh
echo "export LD_LIBRARY_PATH=/usr/local/magma/lib:\$LD_LIBRARY_PATH" >> /etc/profile.d/magma.sh
echo "export PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:/usr/local/magma/lib/pkgconfig" >> /etc/profile.d/magma.sh
