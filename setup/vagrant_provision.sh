#!/bin/bash
set -exuo pipefail
DIR="/vagrant/setup"
/bin/sudo bash "$DIR/make_directories.sh" --dev
/bin/sudo bash "$DIR/install_dependencies.sh" --dev
/bin/sudo bash "$DIR/install_mxlib.sh" --dev
/bin/sudo bash "$DIR/set_permissions.sh"
# Create or replace symlink to sources so we develop on the host machine's copy
# (unlike prod, where we install a new clone of the repo to this location)
ln -nfs /vagrant /opt/MagAOX/source/MagAOX
usermod -G magaox,magaox-dev vagrant
/bin/sudo -u vagrant bash "$DIR/install_MagAOX.sh"
echo "Finished!"
