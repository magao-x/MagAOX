#!/bin/bash
# If not started as root, sudo yourself
if [[ "$EUID" != 0 ]]; then
    sudo bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

MAMBAFORGE_VERSION="23.1.0-4"
MAMBAFORGE_INSTALLER="Mambaforge-$MAMBAFORGE_VERSION-Linux-$(uname -p).sh"
MAMBAFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/$MAMBAFORGE_VERSION/$MAMBAFORGE_INSTALLER"
#
# conda
#
cd /opt/MagAOX/vendor
if [[ ! -d /opt/conda ]]; then
    _cached_fetch "$MAMBAFORGE_URL" $MAMBAFORGE_INSTALLER
    bash $MAMBAFORGE_INSTALLER -b -p /opt/conda
	# Ensure magaox-dev can write to /opt/conda or env creation will fail
	chown -R :$instrument_dev_group /opt/conda
    # set group and permissions such that only magaox-dev has write access
    chmod -R g=rwX /opt/conda
    find /opt/conda -type d -exec sudo chmod g+rwxs {} \;
    # Set environment variables for conda
    cat << 'EOF' | tee /etc/profile.d/conda.sh
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
    CONDA_CHANGEPS1=false conda activate base
else
    \export PATH="/opt/conda/bin:$PATH"
fi
EOF
    cat << 'EOF' | tee /opt/conda/.condarc
channels:
  - conda-forge
  - defaults
changeps1: false
channel_priority: flexible
disallowed_packages: [ qt ]
EOF
fi
