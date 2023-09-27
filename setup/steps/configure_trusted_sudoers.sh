#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -o pipefail

# Defines $ID and $VERSION_ID so we can detect which distribution we're on
source /etc/os-release

scratchFile=/tmp/sudoers_trusted
targetFile=/etc/sudoers.d/trusted

echo '# file automatically created by configure_trusted_sudoers.sh, do not edit' > $scratchFile || exit_error "Could not create $scratchFile"

if [[ $ID == rocky || $ID == centos ]]; then
    echo "User_Alias TRUSTED = %wheel" > $scratchFile
else if [[ $ID == ubuntu ]]; then
    echo "User_Alias TRUSTED = %sudo" > $scratchFile
else
    exit_error "Got ID=$ID, only know rocky, centos, and ubuntu"
fi

cat <<'HERE' | tee -a $scratchFile
Defaults:TRUSTED !env_reset
Defaults:TRUSTED !secure_path
HERE

visudo -cf $scratchFile || exit_error "visudo syntax check failed on /tmp/sudoers_xsup"
sudo mv $scratchFile $targetFile || exit_error "Could not install drop-in file to $targetFile"
