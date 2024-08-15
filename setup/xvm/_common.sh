#!/usr/bin/env bash
if [[ -z $vmArch ]]; then
    echo "Set vmArch environment variable to aarch64 or x86_64"
    exit 1
fi
rockyVersion=${rockyVersion:-9.4}
export rockyVersion
