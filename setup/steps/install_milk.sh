#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -eo pipefail

COMMIT_ISH=dev
orgname=milk-org
reponame=milk
parentdir=/opt/MagAOX/source
clone_or_update_and_cd $orgname $reponame $parentdir
git checkout $COMMIT_ISH

bash -x ./fetch_cacao_dev.sh

mkdir -p _build
cd _build
cmake ..
make
sudo make install

milkSuffix=bin/milk
milkBinary=$(grep -e "${milkSuffix}$" ./install_manifest.txt)
milkBinPath=${milkBinary/${milkSuffix}/bin}

if [[ $(which milk) != $milkBinary ]]; then
    existingMilk=$(which milk)
    log_warn "Found existing milk binary at $existingMilk"
fi

echo "export PATH=\"\$PATH:${milkBinPath}\"" | sudo tee /etc/profile.d/milk-bin-path.sh