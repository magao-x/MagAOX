#!/bin/bash
set -exuo pipefail
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
envswitch=${1:---prod}

if [[ "$envswitch" == "--dev" ]]; then
  ENV=dev
elif [[ "$envswitch" == "--prod" ]]; then
  ENV=prod
else
  cat <<'HERE'
Usage: provision_as_user.sh [--dev] [--prod]
Builds mxLib and any other first-party software. Note: this is not intended
to replace the MagAOX software build system itself, only build dependencies.

  --prod  (default) Set up for production (i.e. default to Intel MKL
          math library)
  --dev   Set up for local development (i.e. default to ATLAS math library)
HERE
  exit 1
fi
#
# mxLib
#
if [[ -d "/opt/MagAOX/source/mxlib" ]]; then
    cd "/opt/MagAOX/source/mxlib"
    git pull
    log "Updated mxlib"
else
    git clone --depth=1 https://github.com/jaredmales/mxlib.git
    log "Cloned a new copy of mxlib"
    cd "/opt/MagAOX/source/mxlib"
fi

export MXMAKEFILE="/opt/MagAOX/source/mxlib/mk/MxApp.mk"
if [[ $ENV == dev ]]; then
  cat << EOF > "/opt/MagAOX/source/mxlib/local/MxApp.mk"
BLAS_INCLUDES = -I/usr/include/atlas-x86_64-base
BLAS_LDFLAGS = -L/usr/lib64/atlas -L/usr/lib64
BLAS_LDLIBS = -ltatlas -lgfortran
EOF
fi
make PREFIX=/usr/local
make install PREFIX=/usr/local
echo "export MXMAKEFILE=\"/opt/MagAOX/source/mxlib/mk/MxApp.mk\"" | tee /etc/profile.d/mxmakefile.sh
