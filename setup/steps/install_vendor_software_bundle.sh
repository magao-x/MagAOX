#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../_common.sh
set -exuo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENDOR_SOFTWARE_BUNDLE=$DIR/vendor_software.tar.gz
if [[ ! -e $VENDOR_SOFTWARE_BUNDLE ]]; then
    echo "Couldn't find vendor software bundle at location $VENDOR_SOFTWARE_BUNDLE"
    echo "(Generate with Box/MagAO-X/Vendor\ Software/generate_bundle.sh)"
    exit 1
fi
BUNDLE_EXTRACT_DIR=/tmp/vendor_software_bundle
MAGAOX_VENDOR_DIR=/opt/MagAOX/vendor
mkdir -p $BUNDLE_EXTRACT_DIR
tar xzf $VENDOR_SOFTWARE_BUNDLE -C $BUNDLE_EXTRACT_DIR
cd $BUNDLE_EXTRACT_DIR
for name in ./*; do
    if [[ -d name ]]; then
        exit 1
    fi
done
