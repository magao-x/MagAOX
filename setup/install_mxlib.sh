#!/bin/bash
source /opt/rh/devtoolset-7/enable
export PATH="/usr/local/bin:$PATH"
set -exuo pipefail
IFS=$'\n\t'
envswitch=${1:---prod}

MXLIBROOT=/opt/MagAOX/source/dependencies/mxlib

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
    echo "Updated mxlib"
else
    git clone --depth=1 https://github.com/jaredmales/mxlib.git "$MXLIBROOT"
    echo "Cloned a new copy of mxlib"
    cd "$MXLIBROOT"
fi

export MXMAKEFILE="$MXLIBROOT/mk/MxApp.mk"
# Populate $MXLIBROOT/local/ with example makefiles:
make setup
if [[ $ENV == dev ]]; then
  cat << EOF > "$MXLIBROOT/local/MxApp.mk"
BLAS_INCLUDES = -I/usr/include/atlas-x86_64-base
BLAS_LDFLAGS = -L/usr/lib64/atlas -L/usr/lib64
BLAS_LDLIBS = -ltatlas -lgfortran
EOF
fi
# Ensure mxlib installs to /usr/local (not $HOME)
if diff "$MXLIBROOT/local/Common.mk" "$MXLIBROOT/local/Common.example.mk"; then
  echo "PREFIX = /usr/local" > "$MXLIBROOT/local/Common.mk"
elif cat "PREFIX = /usr/local" | diff "$MXLIBROOT/local/Common.mk" -; then
  echo "PREFIX already set to /usr/local"
else
  echo "Unexpected modifications in $MXLIBROOT/local/Common.mk! Aborting." >&2
fi
make
make install
# Sanity check: make sure gengithead.sh is available systemwide in /usr/local
gengithead.sh ./ ./include/mxlib_uncomp_version.h MXLIB_UNCOMP
# Ensure all users get $MXMAKEFILE pointing to this install by default
echo "export MXMAKEFILE=\"$MXLIBROOT/mk/MxApp.mk\"" | tee /etc/profile.d/mxmakefile.sh
