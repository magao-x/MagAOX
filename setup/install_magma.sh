#!/bin/bash
set -exuo pipefail
cd $DEPSROOT
MAGMA_VERSION=2.5.1-alpha1
MAGMA_FOLDER=magma-$MAGMA_VERSION
if [[ ! -d $MAGMA_FOLDER ]]; then
    curl -L http://icl.utk.edu/projectsfiles/magma/downloads/$MAGMA_VERSION.tar.gz | tar xvz
fi
cd $MAGMA_FOLDER
cp -n make.inc-examples/make.inc.mkl-gcc ./make.inc
# Limit target architecture to Pascal to save some compilation time
if ! grep "GPU_TARGET = Pascal" make.inc; then
  echo "GPU_TARGET = Pascal" >> make.inc
fi
make -j 32
make install
echo "# Configure MAGMA library environment variables (do not edit, see /opt/MagAOX/source/MagAOX/setup/install_magma.sh)" > /etc/profile.d/magma.sh
echo "export LD_LIBRARY_PATH=/usr/local/magma/lib:\$LD_LIBRARY_PATH" >> /etc/profile.d/magma.sh
echo "export PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:/usr/local/magma/lib/pkgconfig/" >> /etc/profile.d/magma.sh
