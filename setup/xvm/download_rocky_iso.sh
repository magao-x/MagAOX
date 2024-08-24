#!/usr/bin/env bash
source ./_common.sh
mkdir -p ./input/iso
cd ./input/iso
if [[ ! -e Rocky-${rockyVersion}-${vmArch}-minimal.iso ]]; then
    curl --no-progress-meter -L https://download.rockylinux.org/pub/rocky/9/isos/${vmArch}/Rocky-${rockyVersion}-${vmArch}-minimal.iso > Rocky-${rockyVersion}-${vmArch}-minimal.iso.part
    mv Rocky-${rockyVersion}-${vmArch}-minimal.iso.part Rocky-${rockyVersion}-${vmArch}-minimal.iso
else
    echo "Rocky Linux ${vmArch} minimal ISO already downloaded."
fi
