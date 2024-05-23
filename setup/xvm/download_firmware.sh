#!/usr/bin/env bash
set -ex
mkdir -p ./input/firmware
cd ./input/firmware
platform=$(uname)
if [[ $(uname -m) == "arm64" ]]; then
    archInfix=aarch64
    archPrefix=AAVMF
else
    archInfix=x86_64
    archPrefix=OVMF
fi
indexPage=https://dl.rockylinux.org/pub/rocky/9/AppStream/${archInfix}/os/Packages/e/
if [[ ! -e ${archPrefix}_CODE.fd ]]; then
    rpmName=$(curl -q $indexPage | grep edk2 | sed -n 's/.*href="\(edk2-'$archInfix'-[^"]*\)".*/\1/p')
    curl -OL ${indexPage}/${rpmName}
    tar xf $rpmName
    cp usr/share/${archPrefix}/${archPrefix}_*.fd ./
fi
rm -rf usr *.rpm