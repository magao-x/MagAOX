#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail


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

mxlibCommonOverrides="/tmp/$(date +"%s")-mxlibCommon.mk"

cat <<'HERE' > $mxlibCommonOverrides
PREFIX = /usr/local
CXXFLAGS += -DMX_OLD_GSL $(shell pkg-config --cflags lapacke)
NEED_CUDA = no
EIGEN_CFLAGS = $(shell pkg-config --cflags eigen3)
USE_BLAS_FROM = openblas
HERE

if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC || $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == ci ]]; then
  echo "INCLUDES += -I/usr/local/cuda/targets/x86_64-linux/include/" >> $mxlibCommonOverrides
fi

source /etc/os-release

if [[ $ID == rocky && $(uname -p) == "aarch64" ]]; then
  echo "CXXFLAGS += -I/usr/include/lapacke" >> $mxlibCommonOverrides
fi

# Ensure mxlib installs to /usr/local (not $HOME)
if diff local/Common.mk local/Common.example.mk; then
  mv $mxlibCommonOverrides local/Common.mk
elif diff $mxlibCommonOverrides local/Common.mk ; then
  echo "mxlib options configured"
else
  echo "Unexpected modifications in $MXLIBROOT/local/Common.mk! Aborting. See $mxlibCommonOverrides for suggested Common.mk" >&2
  exit 1
fi
make || exit 1
sudo -E make install || exit 1
# Sanity check: make sure gengithead.sh is available systemwide in /usr/local/bin
gengithead.sh ./ ./include/mxlib_uncomp_version.h MXLIB_UNCOMP || exit 1
# Ensure all users get $MXMAKEFILE pointing to this install by default
echo "export MXMAKEFILE=\"$MXLIBROOT/mk/MxApp.mk\"" | sudo tee /etc/profile.d/mxmakefile.sh || exit 1
