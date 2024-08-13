#!/usr/bin/env bash
rockyVersion=${rockyVersion:-9.4}
if [[ $(uname -p) == arm ]]; then
    rockyArch=${rockyArch:-aarch64}
    qemuArch=aarch64
else
    rockyArch=${rockyArch:-x86_64}
    qemuArch=x86_64
fi
export rockyVersion rockyArch qemuArch