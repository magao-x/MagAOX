#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail


MXLIBROOT=/opt/MagAOX/source/mxlib

#
# mxLib
#

MXLIB_COMMIT_ISH=magaox
orgname=jaredmales
reponame=mxlib
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir


if [[ -d "$MXLIBROOT" ]]; then
    cd "$MXLIBROOT"
    git pull
    echo "Updated mxlib"
else
    # TODO: use _common checkout function
    git clone https://github.com/jaredmales/mxlib.git "$MXLIBROOT"
    echo "Cloned a new copy of mxlib"
    cd "$MXLIBROOT"
fi

git config core.sharedRepository group
git checkout $MXLIB_COMMIT_ISH
export MXMAKEFILE="$MXLIBROOT/mk/MxApp.mk"
# Populate $MXLIBROOT/local/ with example makefiles:
make setup

cat <<'HERE' > /tmp/mxlibCommon.mk
PREFIX = /usr/local
CXXFLAGS += -DMX_OLD_GSL
INCLUDES += -I/usr/local/cuda-11.1/targets/x86_64-linux/include/
NEED_CUDA = no
EIGEN_CFLAGS = 
HERE

# Ensure mxlib installs to /usr/local (not $HOME)
if diff local/Common.mk local/Common.example.mk; then
  mv /tmp/mxlibCommon.mk local/Common.mk
elif diff /tmp/mxlibCommon.mk local/Common.mk ; then
  echo "mxlib options configured"
else
  echo "Unexpected modifications in $MXLIBROOT/local/Common.mk! Aborting." >&2
  exit 1
fi
make
sudo make install
# Sanity check: make sure gengithead.sh is available systemwide in /usr/local/bin
gengithead.sh ./ ./include/mxlib_uncomp_version.h MXLIB_UNCOMP
# Ensure all users get $MXMAKEFILE pointing to this install by default
echo "export MXMAKEFILE=\"$MXLIBROOT/mk/MxApp.mk\"" | sudo tee /etc/profile.d/mxmakefile.sh
