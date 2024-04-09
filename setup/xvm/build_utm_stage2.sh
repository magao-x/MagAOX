#!/usr/bin/env bash
echo "Starting up the VM for MagAO-X software installation..."
qemu-system-aarch64 \
    -name xvm \
    -netdev user,id=user.0,hostfwd=tcp:127.0.0.1:2201-:22 \
    -device virtio-keyboard-pci -device virtio-mouse-pci \
    -smp 4 \
    -machine type=virt$accelFlag \
    -cpu $cpuType \
    -drive if=pflash,format=raw,id=ovmf_code,readonly=on,file=./input/firmware/AAVMF_CODE.fd \
    -drive if=pflash,format=raw,id=ovmf_vars,file=./input/firmware/AAVMF_VARS.fd \
    -drive file=output/xvm.qcow2,format=qcow2 \
    -device virtio-gpu-pci \
    -device virtio-net-pci,netdev=user.0 \
    -m 4096M \
    -display none \
&
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
tar -cJvf ./MagAO-X.utm.tar.xz ./MagAO-X.utm xvm_key xvm_key.pub
