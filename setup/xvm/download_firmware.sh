#!/usr/bin/env bash
mkdir -p ./input/firmware
cd ./input/firmware
platform=$(uname)
if [[ ! -e AAVMF_CODE.fd ]]; then
    curl -OL https://dl.rockylinux.org/pub/rocky/9/AppStream/aarch64/os/Packages/e/edk2-aarch64-20230524-4.el9_3.2.noarch.rpm
    if [[ $platform == "Darwin" ]]; then
        tar xvf edk2-aarch64-20230524-4.el9_3.2.noarch.rpm
    else
        rpm2cpio edk2-aarch64-20230524-4.el9_3.2.noarch.rpm | cpio -idmv
    fi
    cp usr/share/AAVMF/AAVMF_*.fd ./
fi
if [[ ! -e OVMF_CODE.fd ]]; then
    curl -OL https://dl.rockylinux.org/pub/rocky/9/AppStream/x86_64/os/Packages/e/edk2-ovmf-20230524-4.el9_3.2.noarch.rpm
    if [[ $platform == "Darwin" ]]; then
        tar xvf edk2-ovmf-20230524-4.el9_3.2.noarch.rpm
    else
        rpm2cpio edk2-ovmf-20230524-4.el9_3.2.noarch.rpm | cpio -idmv
    fi
    cp usr/share/edk2/ovmf/OVMF_*.fd ./
fi
rm -rf usr *.rpm