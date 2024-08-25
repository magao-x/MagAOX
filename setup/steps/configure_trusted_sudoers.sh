#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -o pipefail

# Defines $ID and $VERSION_ID so we can detect which distribution we're on
source /etc/os-release

scratchFile=/tmp/sudoers_trusted
targetFile=/etc/sudoers.d/trusted

echo '# file automatically created by configure_trusted_sudoers.sh, do not edit' > $scratchFile || exit_with_error "Could not create $scratchFile"

if [[ $ID == rocky ]]; then
    echo "User_Alias TRUSTED = %wheel" > $scratchFile
elif [[ $ID == ubuntu ]]; then
    echo "User_Alias TRUSTED = %sudo" > $scratchFile
else
    exit_with_error "Got ID=$ID, only know rocky and ubuntu"
fi

cat <<'HERE' | tee -a $scratchFile
Defaults:TRUSTED !env_reset
Defaults:TRUSTED !secure_path
HERE

visudo -cf $scratchFile || exit_with_error "visudo syntax check failed on /tmp/sudoers_xsup"
sudo install \
    --owner=root \
    --group=root \
    --mode=u=r--g=r--o=--- \
    $scratchFile \
    $targetFile \
|| exit_with_error "Could not install drop-in file to $targetFile"
