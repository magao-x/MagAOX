#!/usr/bin/env bash
set -x
mkdir -p ./input/firmware || exit 1
cd ./input/firmware

indexPage=https://dl.rockylinux.org/pub/rocky/9/AppStream/${vmArch}/os/Packages/e/

rpmName=$(curl -q $indexPage | grep edk2 | sed -n 's/.*href="\(edk2-[^"]*\)".*/\1/p')
curl -OL ${indexPage}/${rpmName} || exit 1
if [[ $(uname -o) == Darwin ]]; then
    tar xvf $rpmName || exit 1
else
    rpm2archive $rpmName || exit 1
    tar xvf $rpmName.tgz || exit 1
fi
