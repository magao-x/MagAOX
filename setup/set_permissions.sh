#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

chown -R root:root /opt/MagAOX

# Operators should be able to access sources
# but modifications should be setGID (+s)
chown -R :magaox-dev /opt/MagAOX/source
chmod -R g=rwX /opt/MagAOX/source
# n.b. can't be recursive because g+s on files means something else
# so we find all directories and individually chmod them:
find /opt/MagAOX/source -type d -exec chmod -v g+s {} \;

# Set logs to writable for non-admin users like xsup
chown -RP xsup:magaox /opt/MagAOX/logs
# Let operators (not in group magaox) read logs but not write:
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/logs

# Set rawimages to writable for non-admin users like xsup
chown -RP xsup:magaox /opt/MagAOX/rawimages
# Let operators (not in group magaox) read rawimages but not write:
chmod -R u=rwX,g=rwX,o=rX /opt/MagAOX/rawimages

# Hide secrets
chmod o-rwx /opt/MagAOX/secrets
chmod g-rwx /opt/MagAOX/secrets
