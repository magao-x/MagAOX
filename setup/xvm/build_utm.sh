#!/usr/bin/env bash
mkdir -p output
# make disk drive image
qemu-img create -f qcow2 output/xvm.qcow2 64G
if [[ ! -e ./input/xvm_key ]]; then
    ssh-keygen -q -t ed25519 -f ./input/xvm_key -N ''
fi
cp -R ./kickstart ./input/
cp ./input/xvm_key.pub ./input/kickstart/authorized_keys
rm -f ./input/oemdrv.{dmg,qcow2}
hdiutil create -srcfolder ./input/kickstart -format UDRO -volname "OEMDRV" -fs "MS-DOS FAT32" ./input/oemdrv.dmg
qemu-img convert -f dmg -O qcow2 ./input/oemdrv.dmg ./input/oemdrv.qcow2
bash download_rocky_iso.sh
bash download_firmware.sh
echo "Starting VM installation process..."
qemu-system-aarch64 \
    -name xvm \
    -cdrom Rocky-9.3-aarch64-minimal.iso \
    -netdev user,id=user.0 \
    -smp 4 \
    -machine type=virt \
    -cpu host \
    -drive if=pflash,format=raw,id=ovmf_code,readonly=on,file=./input/firmware/AAVMF_CODE.fd \
    -drive if=pflash,format=raw,id=ovmf_vars,file=./input/firmware/AAVMF_VARS.fd \
    -drive file=output/xvm.qcow2,format=qcow2 \
    -drive file=input/oemdrv.qcow2,format=qcow2 \
    -device virtio-gpu-pci \
    -device virtio-net-pci,netdev=user.0 \
    -boot c \
    -m 4096M \
|| exit 1
    # -display cocoa \
    # -serial stdio \
echo "Created VM and installed Rocky Linux 9.3 with KDE."
echo "Starting up the VM for MagAO-X software installation..."
qemu-system-aarch64 \
    -name xvm \
    -netdev user,id=user.0,hostfwd=tcp:127.0.0.1:2201-:22 \
    -device virtio-keyboard-pci -device virtio-mouse-pci \
    -smp 4 \
    -machine type=virt \
    -cpu host \
    -drive if=pflash,format=raw,id=ovmf_code,readonly=on,file=./input/firmware/AAVMF_CODE.fd \
    -drive if=pflash,format=raw,id=ovmf_vars,file=./input/firmware/AAVMF_VARS.fd \
    -drive file=output/xvm.qcow2,format=qcow2 \
    -device virtio-gpu-pci \
    -device virtio-net-pci,netdev=user.0 \
    -m 4096M \
&
    # -display cocoa \
    # -serial stdio \
    # -device virtio-serial \
    # -chardev socket,path=/tmp/qga.sock,server=on,wait=off,id=qga0 \
    # -device virtserialport,chardev=qga0,name=org.qemu.guest_agent.0 \
sleep 30
ssh -p 2201 -o "UserKnownHostsFile /dev/null" -o "StrictHostKeyChecking=no" -i ./input/xvm_key xdev@localhost 'bash -s' < ./bootstrap_magao-x.sh
wait
cp -R ./utm ./output/MagAO-X.utm
cd ./output
mv ./xvm.qcow2 ./MagAO-X.utm/Data/xvm.qcow2
tar -cJvf ./MagAO-X.utm.tar.xz ./MagAO-X.utm
