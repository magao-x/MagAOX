#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

TARGET_ENV=$1

if [[ "$TARGET_ENV" == "vagrant" ]]; then
  CMAKE_FLAGS=""
elif [[ "$TARGET_ENV" == "instrument" ]]; then
  CMAKE_FLAGS="-DUSE_CUDA=YES -DUSE_MAGMA=YES"
else
  echo "Unknown TARGET_ENV passed as argument 1"
  exit 1
fi

if [[ ! -d ./cacao ]]; then
    git clone --recursive https://github.com/cacao-org/cacao cacao
    cd ./cacao
else
    cd ./cacao
    git fetch
fi
CACAO_ABSPATH=$PWD
git submodule update --recursive --remote
git checkout dev
git submodule foreach git checkout dev
git submodule foreach git pull
git fetch --all --tags --prune
git submodule foreach "git fetch --all --tags --prune"
git submodule update

mkdir -p _build
cd _build
cmake3 ../ $CMAKE_FLAGS
make
/bin/sudo make install
if [[ ! -e /usr/local/bin/milk ]]; then
    sudo ln -s /usr/local/bin/cacao /usr/local/bin/milk
fi
echo "export PATH=\$PATH:$CACAO_ABSPATH/src/CommandLineInterface/scripts" | sudo tee /etc/profile.d/cacao_scripts.sh
