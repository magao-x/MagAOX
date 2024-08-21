#!/usr/bin/env bash
if [[ -z $vmArch ]]; then
    echo "Set vmArch environment variable to aarch64 or x86_64"
    exit 1
fi
if [[ $(uname -p) == "arm" && $CI != "true" ]]; then
    qemuMachineFlags="-machine type=virt,highmem=on -cpu host"
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

nCpus=3
ramMB=8192

if [[ $CI == true ]]; then
    qemuAccelFlags="-accel tcg,thread=multi"
else
    qemuAccelFlags="-accel kvm -accel hvf -accel tcg,thread=multi"
fi

qemuSystemCommand="qemu-system-${vmArch} \
    -name xvm \
    -netdev user,id=user.0,hostfwd=tcp:127.0.0.1:2201-:22 \
    -device virtio-keyboard-pci -device virtio-mouse-pci \
    -smp $nCpus \
    $qemuAccelFlags \
    $qemuMachineFlags \
    -drive if=pflash,format=raw,id=ovmf_code,readonly=on,file=./output/firmware_code.fd \
    -drive if=pflash,format=raw,id=ovmf_vars,file=./output/firmware_vars.fd \
    -drive file=output/xvm.qcow2,format=qcow2 \
    -device virtio-gpu-pci \
    -device virtio-net-pci,netdev=user.0 \
    -boot c \
    -m ${ramMB}M \
    $ioFlag "
export qemuSystemCommand

export rockyVersion=${rockyVersion:-9.4}

function updateGuestMagAOXCheckout() {
    echo "Syncing ~/MagAOX/ in guest..."
    rsync --progress -a --exclude xvm/output --exclude xvm/input -e 'ssh -p 2201 -o "UserKnownHostsFile /dev/null" -o "StrictHostKeyChecking=no" -i ./output/xvm_key' ../../ xdev@localhost:MagAOX/
    echo "Finished updating ~/MagAOX/ in guest"
}
