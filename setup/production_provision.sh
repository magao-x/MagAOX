#!/bin/bash
set -exuo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set +e; groups | grep magaox-dev; set -e
not_in_group=$?
if [[ "$EUID" == 0 || $not_in_group != 0 ]]; then
  echo "This script should be run as a normal user"
  echo "in the magaox-dev group with sudo access, not root."
  echo "Run $DIR/setup_users_and_groups.sh first."
  exit 1
fi
# Prompt for sudo authentication
/bin/sudo -v
# Keep the sudo timestamp updated until this script exits
while true; do /bin/sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

/bin/sudo bash "$DIR/make_directories.sh"
/bin/sudo bash "$DIR/install_mkl.sh"
/bin/sudo bash "$DIR/install_dependencies.sh"
/bin/sudo bash "$DIR/install_mxlib.sh"
/bin/sudo bash "$DIR/set_permissions.sh"

if [[ $DIR != /opt/MagAOX/source/MagAOX ]]; then
    if [[ ! -e /opt/MagAOX/source/MagAOX ]]; then
        echo "Cloning new copy of MagAOX codebase"
        git clone $(dirname $DIR) /opt/MagAOX/source/MagAOX
    fi
    cd /opt/MagAOX/source/MagAOX
    git remote remove origin
    git remote add origin https://github.com/magao-x/MagAOX.git
    git fetch
    git branch -u origin/master master
    echo "In the future, you can run this script from /opt/MagAOX/source/MagAOX/setup"
else
    echo "Running from clone located at $DIR, nothing to do for cloning step"
fi
# The last step should work as whatever user is installing, provided
# they are a member of magaox-dev and they have sudo access to install to
# /usr/local. Building as root would leave intermediate build products
# owned by root, which we probably don't want.
bash "$DIR/install_MagAOX.sh"
echo "Finished!"
