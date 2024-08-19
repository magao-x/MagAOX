#!/usr/bin/env bash
set -ex
mkdir -p ./input/firmware || exit 1
cd ./input/firmware || exit 1

if [[ $vmArch == aarch64 ]]; then
    archInfix=aarch64
    archFolder=aavmf
    archPrefix=AAVMF
else
    archInfix=x86_64
    archFolder=ovmf
    archPrefix=OVMF
fi

indexPage=https://dl.rockylinux.org/pub/rocky/9/AppStream/${archInfix}/os/Packages/e/
if [[ ! -e ${archPrefix}_CODE.fd ]]; then
    rpmName=$(curl -q $indexPage | grep edk2 | sed -n 's/.*href="\(edk2-[^"]*\)".*/\1/p')
    curl -OL ${indexPage}/${rpmName} || exit 1
    if [[ $(uname -o) == Darwin ]]; then
        tar xvf $rpmName || exit 1
    else
        rpm2archive $rpmName || exit 1
        tar xvf $rpmName.tgz || exit 1
    fi
    cp -v usr/share/${archFolder}/${archPrefix}_*.fd ./ || exit 1
fi
rm -rf usr *.rpm