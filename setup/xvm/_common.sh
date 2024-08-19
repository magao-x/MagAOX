#!/usr/bin/env bash
if [[ -z $vmArch ]]; then
    echo "Set vmArch environment variable to aarch64 or x86_64"
    exit 1
fi
if [[ $(uname -p) == "arm" && $CI != "true" ]]; then
    qemuMachineFlags="-machine type=virt,highmem=on,accel=hvf:kvm -cpu host"
elif [[ $(uname -p) == "arm" ]]; then
    qemuMachineFlags="-machine type=virt -cpu max"
elif [[ $(uname -p) == "x86_64" ]]; then
    qemuMachineFlags=""
fi
export qemuMachineFlags

qemuDisplay=${qemuDisplay:-}
if [[ ! -z $qemuDisplay ]]; then
    ioFlag="-display $qemuDisplay"
else
    ioFlag='-serial stdio -display none'
fi
export ioFlag

export rockyVersion=${rockyVersion:-9.4}
