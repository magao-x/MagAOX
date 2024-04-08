#!/usr/bin/env bash
cd ./input/
if [ ! -e Rocky-9.3-aarch64-minimal.iso ]; then
    curl -OL https://download.rockylinux.org/pub/rocky/9/isos/aarch64/Rocky-9.3-aarch64-minimal.iso
fi