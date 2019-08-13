#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail
cd alpao
patch -Np1 < alpao_build_fix.patch || true
# option 2 - "Install ASDK and Interface Corp. PEX-292144 support"
echo 2 | sudo bash Linux/InstallASDK.sh
# The Alpao installer (and PEX sub-installer) doesn't explicitly set permissions on its libs
for libFilename in libgpg2x72c.so.2.2.5 libgpgconf.so.1.5.6 libait_pex292144.so libasdk.so; do
    chmod -v u=rwx,g=rx,o=rx /usr/lib/$libFilename
    chmod -v u=rwx,g=rx,o=rx /usr/lib64/$libFilename
done
