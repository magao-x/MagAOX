#!/usr/bin/env bash
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
cp ./input/firmware/AAVMF_VARS.fd ./output/AAVMF_VARS.fd
if [[ $(uname -p) == "arm" && $CI != "true" ]]; then
    cpuType="host"
    accelFlag=",highmem=on,accel=hvf:kvm"
else
    cpuType="max"
    accelFlag=""
fi
echo "Starting VM installation process..."
qemu-system-aarch64 \
    -name xvm \
    -cdrom ./input/iso/Rocky-9.3-aarch64-minimal.iso \
    -netdev user,id=user.0 \
    -device virtio-keyboard-pci -device virtio-mouse-pci \
    -smp 4 \
    -machine type=virt$accelFlag \
    -cpu $cpuType \
    -drive if=pflash,format=raw,id=ovmf_code,readonly=on,file=./input/firmware/AAVMF_CODE.fd \
    -drive if=pflash,format=raw,id=ovmf_vars,file=./output/AAVMF_VARS.fd \
    -drive file=output/xvm.qcow2,format=qcow2 \
    -drive file=input/oemdrv.qcow2,format=qcow2 \
    -device virtio-gpu-pci \
    -device virtio-net-pci,netdev=user.0 \
    -boot c \
    -m 8192M \
    -display none \
|| exit 1
cp -v ./output/xvm.qcow2 ./output/xvm_stage1.qcow2
echo "Created VM and installed Rocky Linux 9.3 with KDE."
