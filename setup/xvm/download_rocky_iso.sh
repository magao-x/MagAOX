#!/usr/bin/env bash
source ./_common.sh
mkdir -p ./input/iso
cd ./input/iso
if [[ ! -e Rocky-${rockyVersion}-${rockyArch}-minimal.iso ]]; then
    curl -L https://download.rockylinux.org/pub/rocky/9/isos/${rockyArch}/Rocky-${rockyVersion}-${rockyArch}-minimal.iso > Rocky-${rockyVersion}-${rockyArch}-minimal.iso.part
    mv Rocky-${rockyVersion}-${rockyArch}-minimal.iso.part Rocky-${rockyVersion}-${rockyArch}-minimal.iso
else
    echo "Rocky Linux ${rockyArch} minimal ISO already downloaded."
fi
