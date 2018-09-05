#!/bin/bash
set -exuo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
/bin/sudo bash "$DIR/install_dependencies.sh"
/bin/sudo bash "$DIR/make_directories.sh"
/bin/sudo bash "$DIR/install_mxlib.sh"
/bin/sudo bash "$DIR/production_clone_MagAOX.sh"
/bin/sudo bash "$DIR/set_permissions.sh"
/bin/sudo bash "$DIR/install_MagAOX.sh"
echo "Finished!"
