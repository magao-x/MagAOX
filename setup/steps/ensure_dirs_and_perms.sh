#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -uo pipefail

# Exit if any of these fail:
set -e

mkdir -pv /opt/MagAOX
mkdir -pv /opt/MagAOX/bin
mkdir -pv /opt/MagAOX/drivers/fifos
mkdir -pv /opt/MagAOX/secrets
mkdir -pv /opt/MagAOX/sys
mkdir -pv /opt/MagAOX/vendor
mkdir -pv /opt/MagAOX/source

mkdir -pv /opt/MagAOX/calib
chown -R root:$instrument_group /opt/MagAOX/calib
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/calib
setgid_all /opt/MagAOX/calib

mkdir -pv /opt/MagAOX/config
chown -R root:$instrument_dev_group /opt/MagAOX/config
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/config
setgid_all /opt/MagAOX/config
mkdir -pv /opt/MagAOX/.cache
chown -R root:$instrument_dev_group /opt/MagAOX/.cache
chmod -R u=rwX,g=rwsX,o=rX /opt/MagAOX/.cache

chown root:root /opt/MagAOX
# n.b. not using -R on *either* chown *or* chmod so we don't clobber setuid binaries
chown root:root /opt/MagAOX/bin
chmod u+rwX,g+rX,o+rX /opt/MagAOX/bin

chown -R root:root /opt/MagAOX/drivers
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/drivers
chown -R root:$instrument_group /opt/MagAOX/drivers/fifos

make_on_data_array logs /opt/MagAOX
make_on_data_array rawimages /opt/MagAOX
make_on_data_array telem /opt/MagAOX

chown -R root:root /opt/MagAOX/secrets
chmod -R u=rwX,g=X,o=X /opt/MagAOX/secrets

chown -R root:$instrument_dev_group /opt/MagAOX/source
# n.b. using + instead of = so we don't clobber setuid binaries
chmod -R u+rwX,g+rwX,o+rX /opt/MagAOX/source
setgid_all /opt/MagAOX/source


chown -R root:root /opt/MagAOX/sys
chmod -R u=rwX,g=rX,o=rX /opt/MagAOX/sys

chown root:$instrument_dev_group /opt/MagAOX/vendor
chmod u=rwX,g=rwX,o=rX /opt/MagAOX/vendor
setgid_all /opt/MagAOX/vendor
