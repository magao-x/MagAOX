#!/bin/bash
set -exuo pipefail
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
envswitch=${1:---prod}

MXLIBROOT=/opt/MagAOX/source/mxlib

if [[ "$envswitch" == "--dev" ]]; then
  ENV=dev
elif [[ "$envswitch" == "--prod" ]]; then
  ENV=prod
else
  cat <<'HERE'
Usage: install_mxlib.sh [--dev] [--prod]
Builds mxLib from the standard location for MagAOX machines
(/opt/MagAOX/source/mxlib) and configures $MXMAKEFILE using
a system-wide /etc/profile.d/ entry.

  --prod  (default) Set up for production (i.e. default to Intel MKL
          math library)
  --dev   Set up for local development (i.e. default to ATLAS math library)
HERE
  exit 1
fi
#
# mxLib
#
if [[ -d "$MXLIBROOT" ]]; then
    cd "$MXLIBROOT"
    git pull
    log "Updated mxlib"
else
    git clone --depth=1 https://github.com/jaredmales/mxlib.git
    log "Cloned a new copy of mxlib"
    cd "$MXLIBROOT"
fi

export MXMAKEFILE="$MXLIBROOT/mk/MxApp.mk"
if [[ $ENV == dev ]]; then
  cat << EOF > "$MXLIBROOT/local/MxApp.mk"
BLAS_INCLUDES = -I/usr/include/atlas-x86_64-base
BLAS_LDFLAGS = -L/usr/lib64/atlas -L/usr/lib64
BLAS_LDLIBS = -ltatlas -lgfortran
EOF
fi
make PREFIX=/usr/local
make install PREFIX=/usr/local
echo "export MXMAKEFILE=\"$MXLIBROOT/mk/MxApp.mk\"" | tee /etc/profile.d/mxmakefile.sh
