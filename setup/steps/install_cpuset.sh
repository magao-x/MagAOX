#!/usr/bin/env bash
if [[ "$EUID" != 0 ]]; then
    echo "Becoming root..."
    sudo -H bash -l $0 "$@"
    exit $?
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

if [[ -e /usr/bin/python ]]; then
    osPython=/usr/bin/python
elif [[ -e /bin/python ]]; then
    osPython=/bin/python
else
    osPython=python
fi

# From https://github.com/lpechacek/cpuset.git
mkdir -p /opt/MagAOX/vendor || exit 1
cd /opt/MagAOX/vendor || exit 1

conda deactivate
_cached_fetch https://github.com/PythonCharmers/python-future/archive/refs/tags/v0.18.3.zip python-future-v0.18.3.zip || exit 1
unzip -o python-future-v0.18.3.zip || exit 1
cd python-future-0.18.3 || exit 1
$osPython setup.py bdist_rpm || exit 1
rpmFile=/opt/MagAOX/vendor/python-future-0.18.3/dist/future-0.18.3-1.noarch.rpm
sudo yum install -y $rpmFile

cd /opt/MagAOX/vendor || exit 1

_cached_fetch https://github.com/lpechacek/cpuset/archive/refs/tags/v1.6.zip cpuset-v1.6.zip || exit 1
unzip -o cpuset-v1.6.zip || exit 1
cd cpuset-1.6/ || exit 1
sudo -H yum install -y python-devel xmlto

$osPython setup.py bdist_rpm || exit 1
rpmFile=/opt/MagAOX/vendor/cpuset-1.6/dist/cpuset-1.6-1.noarch.rpm
sudo yum install -y $rpmFile
if [[ -e /etc/profile.d/cgroups1_cpuset_mountpoint.sh ]]; then
    source /etc/profile.d/cgroups1_cpuset_mountpoint.sh
else
    CGROUPS1_CPUSET_MOUNTPOINT=/opt/MagAOX/cpuset
fi
echo "mountpoint = $CGROUPS1_CPUSET_MOUNTPOINT" | sudo -H tee /etc/cset.conf || exit 1
cset --help || exit_with_error "Could not run cset"