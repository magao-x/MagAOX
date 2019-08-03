#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

# mxlib dependencies available as repo packages
yum -y install \
  boost-devel \
  gsl \
  gsl-devel

MXLIBROOT=/opt/MagAOX/source/mxlib

#
# mxLib
#
if [[ -d "$MXLIBROOT" ]]; then
    cd "$MXLIBROOT"
    git pull
    echo "Updated mxlib"
else
    git clone --depth=1 https://github.com/jaredmales/mxlib.git "$MXLIBROOT"
    echo "Cloned a new copy of mxlib"
    cd "$MXLIBROOT"
fi
git config core.sharedRepository group
export MXMAKEFILE="$MXLIBROOT/mk/MxApp.mk"
# Populate $MXLIBROOT/local/ with example makefiles:
make setup

# Ensure mxlib installs to /usr/local (not $HOME)
if diff local/Common.mk local/Common.example.mk; then
  echo "PREFIX = /usr/local" > local/Common.mk
elif echo "PREFIX = /usr/local" | diff local/Common.mk -; then
  echo "PREFIX already set to /usr/local"
else
  echo "Unexpected modifications in $MXLIBROOT/local/Common.mk! Aborting." >&2
  exit 1
fi
make
make install
# Sanity check: make sure gengithead.sh is available systemwide in /usr/local/bin
gengithead.sh ./ ./include/mxlib_uncomp_version.h MXLIB_UNCOMP
# Ensure all users get $MXMAKEFILE pointing to this install by default
echo "export MXMAKEFILE=\"$MXLIBROOT/mk/MxApp.mk\"" | tee /etc/profile.d/mxmakefile.sh
