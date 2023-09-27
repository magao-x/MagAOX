#!/usr/bin/env bash
if [[ "$EUID" != 0 ]]; then
    echo "Becoming root..."
    sudo bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -exuo pipefail

if [[ -e /usr/bin/python ]]; then
    osPython=/usr/bin/python
elif [[ -e /bin/python ]]; then
    osPython=/bin/python
else
    osPython=python
fi

# From https://github.com/lpechacek/cpuset.git
mkdir -p /opt/MagAOX/vendor
cd /opt/MagAOX/vendor

conda deactivate || true
_cached_fetch https://github.com/PythonCharmers/python-future/archive/refs/tags/v0.18.3.zip python-future-v0.18.3.zip
unzip -o python-future-v0.18.3.zip
cd python-future-0.18.3
$osPython setup.py bdist_rpm
rpmFile=/opt/MagAOX/vendor/python-future-0.18.3/dist/future-0.18.3-1.noarch.rpm
sudo yum install -y $rpmFile || true

_cached_fetch https://github.com/lpechacek/cpuset/archive/refs/tags/v1.6.zip cpuset-v1.6.zip
unzip -o cpuset-v1.6.zip
cd cpuset-1.6/
sudo yum install -y python-devel xmlto

$osPython setup.py bdist_rpm
rpmFile=/opt/MagAOX/vendor/cpuset-1.6/dist/cpuset-1.6-1.noarch.rpm
sudo yum install -y $rpmFile || true
source /etc/profile.d/cgroups1_cpuset_mountpoint.sh
cat <<"HERE" | sudo tee /etc/cset.conf || exit 1
mountpoint = $CGROUPS1_CPUSET_MOUNTPOINT
HERE
cset --help || exit_error "Could not run cset"