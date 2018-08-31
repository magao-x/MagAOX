#!/bin/bash
set -exuo pipefail
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#
# mxLib
#
if [[ -d "$HOME/mxlib" ]]; then
    cd "$HOME/mxlib"
    git pull
    log "Updated mxlib"
else
    git clone --depth=1 https://github.com/jaredmales/mxlib.git
    log "Cloned a new copy of mxlib"
    cd "$HOME/mxlib"
fi
MXMAKEFILE="$HOME/mxlib/mk/MxApp.mk"
export MXMAKEFILE
cat << EOF > "$HOME/mxlib/local/MxApp.mk"
BLAS_INCLUDES = -I/usr/include/atlas-x86_64-base
BLAS_LDFLAGS = -L/usr/lib64/atlas -L/usr/lib64
BLAS_LDLIBS = -ltatlas -lgfortran
EOF
make PREFIX=/usr/local
make install PREFIX=/usr/local
cd ..

#
# aoSystem (demo, remove later)
#
MXMAKEFILE=$HOME/mxlib/mk/MxApp.mk
if [[ -d ./aoSystem ]]; then
    cd aoSystem
    git pull
    log "Updated aoSystem"
else
    git clone https://github.com/jaredmales/aoSystem.git
    cd aoSystem
fi
make -B -f $MXMAKEFILE aoSystem USE_BLAS_FROM=ATLAS