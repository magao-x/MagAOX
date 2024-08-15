#!/usr/bin/env bash
source ./_common.sh
set -x
if [[ -e ./output/xvm_stage1.qcow2 ]]; then
    echo "Stage one image populated from cache. Skipping stage one."
    exit 0
fi
mkdir -p output input
# make disk drive image
qemu-img create -f qcow2 output/xvm.qcow2 64G
# make ssh key pair
if [[ ! -e ./output/xvm_key ]]; then
    ssh-keygen -q -t ed25519 -f ./output/xvm_key -N ''
fi
# create oemdrv disk image for kickstart files and key
bash create_oemdrv.sh
bash download_rocky_iso.sh
bash download_firmware.sh
if [[ $vmArch == aarch64 ]]; then
    cp ./input/firmware/AAVMF_VARS.fd ./output/firmware_vars.fd
    cp ./input/firmware/AAVMF_CODE.fd ./output/firmware_code.fd
else
    cp ./input/firmware/OVMF_VARS.fd ./output/firmware_vars.fd
    cp ./input/firmware/OVMF_CODE.fd ./output/firmware_code.fd
fi
if [[ $(uname -p) == "arm" && $CI != "true" ]]; then
    cpuType="host"
    accelFlag=",highmem=on,accel=hvf:kvm"
else
    cpuType="max"
    accelFlag=""
fi
echo "Starting VM installation process..."
qemuDisplay=${qemuDisplay:-0}
if [[ $qemuDisplay == 0 ]]; then
    ioFlag="-serial stdio"
else
    ioFlag="-display $qemuDisplay"
fi
qemu-system-${vmArch} \
    -name xvm \
    -cdrom ./input/iso/Rocky-${rockyVersion}-${vmArch}-minimal.iso \
    -netdev user,id=user.0 \
    -device virtio-keyboard-pci -device virtio-mouse-pci \
    -smp 3 \
    -machine type=virt$accelFlag \
    -cpu $cpuType \
    -drive if=pflash,format=raw,id=ovmf_code,readonly=on,file=./output/firmware_code.fd \
    -drive if=pflash,format=raw,id=ovmf_vars,file=./output/firmware_vars.fd \
    -drive file=output/xvm.qcow2,format=qcow2 \
    -drive file=input/oemdrv.qcow2,format=qcow2 \
    -device virtio-gpu-pci \
    -device virtio-net-pci,netdev=user.0 \
    -boot c \
    -m 8192M \
    $ioFlag \
    -serial stdio \
|| exit 1
mv -v ./output/xvm.qcow2 ./output/xvm_stage1.qcow2
echo "Created VM and installed Rocky Linux"
