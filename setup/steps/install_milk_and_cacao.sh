#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

COMMIT_ISH=dev
orgname=milk-org
reponame=milk
parentdir=/opt/MagAOX/source
noUpdatesThanks=$parentdir/$reponame/NO_UPDATES_THANKS

if [[ -e $noUpdatesThanks ]]; then
  log_info "Lock file at $noUpdatesThanks indicates it's all good, no updates thanks"
  cd $parentdir/$reponame || exit 1
else
  clone_or_update_and_cd $orgname $reponame $parentdir || exit 1
  git checkout $COMMIT_ISH || exit 1
  bash -x ./fetch_cacao_dev.sh || exit 1
fi
sudo -H rm -rf _build src/config.h src/milk_config.h || exit 1
mkdir -p _build || exit 1
cd _build || exit 1

pythonExe=/opt/conda/bin/python

milkCmakeArgs="-DCMAKE_INSTALL_PREFIX=/usr/local -Dbuild_python_module=ON -DPYTHON_EXECUTABLE=${pythonExe}"

if [[ $MAGAOX_ROLE == TIC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == AOC ]]; then
    milkCmakeArgs="-DUSE_CUDA=ON ${milkCmakeArgs}"
fi

cmake .. $milkCmakeArgs || exit 1
numCpus=$(nproc)
make -j $((numCpus / 2)) || exit 1
sudo make install || exit 1

sudo -H $pythonExe -m pip install ../src/ImageStreamIO/ || exit 1
$pythonExe -c 'import ImageStreamIOWrap' || exit 1

milkSuffix=bin/milk
milkBinary=$(grep -e "${milkSuffix}$" ./install_manifest.txt)
milkPath=${milkBinary/${milkSuffix}/}

if command -v milk; then
    log_warn "Found existing milk binary at $(command -v milk)"
fi
link_if_necessary $milkPath /usr/local/milk || exit 1
echo "/usr/local/milk/lib" | sudo -H tee /etc/ld.so.conf.d/milk.conf || exit 1
sudo -H ldconfig || exit 1
echo "export PATH=\"\$PATH:/usr/local/milk/bin\"" | sudo tee /etc/profile.d/milk.sh || exit 1
echo "export PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:/usr/local/milk/lib/pkgconfig" | sudo tee -a /etc/profile.d/milk.sh || exit 1
echo "export MILK_SHM_DIR=/milk/shm" | sudo tee -a /etc/profile.d/milk.sh || exit 1
echo "export MILK_ROOT=/opt/MagAOX/source/milk" | sudo tee -a /etc/profile.d/milk.sh || exit 1
echo "export MILK_INSTALLDIR=/usr/local/milk" | sudo tee -a /etc/profile.d/milk.sh || exit 1

sudo mkdir -p /milk/shm || exit 1
if [[ $MAGAOX_ROLE != ci && $MAGAOX_ROLE != container ]]; then
  if ! grep -q "/milk/shm" /etc/fstab; then
    echo "tmpfs /milk/shm tmpfs rw,nosuid,nodev" | sudo tee -a /etc/fstab || exit 1
    log_success "Created /milk/shm tmpfs mountpoint"
    sudo mount /milk/shm || exit 1
    log_success "Mounted /milk/shm"
  else
    log_info "Skipping /milk/shm mount setup"
  fi
fi
if [[ $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == RTC ]]; then
  clone_or_update_and_cd magao-x "cacao-${MAGAOX_ROLE,,}" /data || exit 1
  link_if_necessary "/data/cacao-${MAGAOX_ROLE,,}" /opt/MagAOX/cacao || exit 1
  sudo install $DIR/../systemd_units/cacao_startup_if_present.service /etc/systemd/system/ || exit 1
  sudo -H systemctl daemon-reload || true
  sudo systemctl enable cacao_startup_if_present.service || true
else
  make_on_data_array "cacao-${MAGAOX_ROLE,,}" /opt/MagAOX || exit 1
  sudo ln -sf "/opt/MagAOX/cacao-${MAGAOX_ROLE,,}" /opt/MagAOX/cacao || exit 1
fi
log_info "Making /opt/MagAOX/cacao/ owned by $instrument_user:$instrument_group"
sudo chown -R $instrument_user:$instrument_group /opt/MagAOX/cacao/ || exit 1
