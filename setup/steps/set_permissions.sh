#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -euo pipefail

function setgid_all() {
    # n.b. can't be recursive because g+s on files means something else
    # so we find all directories and individually chmod them:
    find $1 -type d -exec chmod g+s {} \;
}

chown root:root /opt/MagAOX

chown -R root:magaox-dev /opt/MagAOX/bin
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/bin
setgid_all /opt/MagAOX/bin

chown -R root:magaox-dev /opt/MagAOX/calib
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/calib
setgid_all /opt/MagAOX/calib

chown -R root:magaox-dev /opt/MagAOX/config
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/config
setgid_all /opt/MagAOX/config

chown -R root:root /opt/MagAOX/drivers
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/drivers

LOGS_DIR=/opt/MagAOX/logs
if [[ -h $LOGS_DIR ]]; then
    LOGS_DIR=$(readlink $LOGS_DIR)
fi
chown -RP xsup:magaox $LOGS_DIR
chmod -R u=rwX,g=rwX,o=rX $LOGS_DIR
setgid_all $LOGS_DIR

RAWIMAGES_DIR=/opt/MagAOX/rawimages
if [[ -h $RAWIMAGES_DIR ]]; then
    RAWIMAGES_DIR=$(readlink $RAWIMAGES_DIR)
fi
chown -RP xsup:magaox $RAWIMAGES_DIR
chmod -R u=rwX,g=rwX,o=rX $RAWIMAGES_DIR
setgid_all $RAWIMAGES_DIR

chown -R root:root /opt/MagAOX/secrets
chmod -R u=rwX,g=,o= /opt/MagAOX/secrets

chown -R root:magaox-dev /opt/MagAOX/source
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/source
setgid_all /opt/MagAOX/source

chown -R root:root /opt/MagAOX/sys
chmod -R u=rwX,g=rX,o=rX /opt/MagAOX/sys

chown -R root:magaox-dev /opt/MagAOX/vendor
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/vendor
setgid_all /opt/MagAOX/vendor
