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
    cp ./input/firmware/usr/share/AAVMF/AAVMF_VARS.fd ./output/firmware_vars.fd
    cp ./input/firmware/usr/share/AAVMF/AAVMF_CODE.fd ./output/firmware_code.fd
else
    cp ./input/firmware/usr/share/edk2/ovmf/OVMF_VARS.fd ./output/firmware_vars.fd
    cp ./input/firmware/usr/share/edk2/ovmf/OVMF_CODE.fd ./output/firmware_code.fd
fi

echo "Starting VM installation process..."
$qemuSystemCommand \
    -cdrom ./input/iso/Rocky-${rockyVersion}-${vmArch}-minimal.iso \
    -drive file=input/oemdrv.qcow2,format=qcow2 \
|| exit 1
mv -v ./output/xvm.qcow2 ./output/xvm_stage1.qcow2
echo "Created VM and installed Rocky Linux"
