#!/bin/bash
set -exuo pipefail
/vagrant/setup/provision_as_root.sh
/vagrant/setup/makeDirs.sh --dev
usermod -G magaox vagrant
/bin/sudo -u vagrant bash /vagrant/setup/provision_as_user.sh
echo "Finished!"
