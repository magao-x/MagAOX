#!/usr/bin/env bash
set -ex
mkdir -p ./input/firmware
cd ./input/firmware

if [[ $vmArch == aarch64 ]]; then
    archInfix=aarch64
    archPrefix=AAVMF
else
    archInfix=x86_64
    archPrefix=OVMF
fi

indexPage=https://dl.rockylinux.org/pub/rocky/9/AppStream/${archInfix}/os/Packages/e/
if [[ ! -e ${archPrefix}_CODE.fd ]]; then
    rpmName=$(curl -q $indexPage | grep edk2 | sed -n 's/.*href="\(edk2-[^"]*\)".*/\1/p')
    curl -OL ${indexPage}/${rpmName}
    if [[ $(uname -o) == Darwin ]]; then
        tar xvf $rpmName
    else
        rpm2archive $rpmName
        tar xvf $rpmName.tgz
    fi
    cp -v usr/share/${archPrefix}/${archPrefix}_*.fd ./
fi
rm -rf usr *.rpm