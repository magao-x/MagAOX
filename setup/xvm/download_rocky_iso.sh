#!/usr/bin/env bash
mkdir -p ./input/iso
cd ./input/iso
ls -la
if [[ ! -e Rocky-9.3-aarch64-minimal.iso ]]; then
    curl -OL https://download.rockylinux.org/pub/rocky/9/isos/aarch64/Rocky-9.3-aarch64-minimal.iso
fi
ls -la