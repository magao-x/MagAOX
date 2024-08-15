#!/usr/bin/env bash
if [[ -z $vmArch ]]; then
    echo "Set vmArch environment variable to aarch64 or x86_64"
    exit 1
fi

if [[ $(uname -p) == "arm" && $CI != "true" ]]; then
    cpuType="host"
    accelFlag=",highmem=on,accel=hvf:kvm"
else
    cpuType="max"
    accelFlag=""
fi
export cpuType accelFlag

qemuDisplay=${qemuDisplay:-}
if [[ ! -z $qemuDisplay ]]; then
    ioFlag="-display $qemuDisplay"
else
    ioFlag='-serial stdio'
fi
export ioFlag

export rockyVersion=${rockyVersion:-9.4}
