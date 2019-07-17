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
    git clone --recursive https://github.com/cacao-org/cacao.git cacao
    cd ./cacao
    git checkout dev
    git submodule foreach git checkout dev
else
    cd ./cacao
fi
CACAO_ABSPATH=$PWD

if [[ ! -d _build ]]; then
    mkdir -p _build
    cd _build
    cmake3 ../ $CMAKE_FLAGS
else
    cd _build
fi
make
/bin/sudo make install
if [[ ! -e /usr/local/bin/milk ]]; then
    sudo ln -s /usr/local/bin/cacao /usr/local/bin/milk
fi
echo "export PATH=\$PATH:$CACAO_ABSPATH/src/CommandLineInterface/scripts" | sudo tee /etc/profile.d/cacao_scripts.sh
