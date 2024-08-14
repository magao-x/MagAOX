#!/usr/bin/env bash
echo "Starting up the VM for MagAO-X software installation..."
source ./_common.sh
if [[ -e ./output/xvm_stage1.qcow2 ]]; then
    cp ./output/xvm_stage1.qcow2 ./output/xvm.qcow2
fi
if [[ $(uname -p) == "arm" && $CI != "true" ]]; then
    cpuType="host"
    accelFlag=",highmem=on,accel=hvf:kvm"
else
    cpuType="max"
    accelFlag=""
fi
qemu-system-${vmArch} \
    -name xvm \
    -netdev user,id=user.0,hostfwd=tcp:127.0.0.1:2201-:22 \
    -device virtio-keyboard-pci -device virtio-mouse-pci \
    -smp 4 \
    -machine type=virt$accelFlag \
    -cpu $cpuType \
    -drive if=pflash,format=raw,id=ovmf_code,readonly=on,file=./output/firmware_code.fd \
    -drive if=pflash,format=raw,id=ovmf_vars,file=./output/firmware_vars.fd \
    -drive file=output/xvm.qcow2,format=qcow2 \
    -device virtio-gpu-pci \
    -device virtio-net-pci,netdev=user.0 \
    -m 8192M \
    -display none \
&
    # -serial stdio \
    # -device virtio-serial \
    # -chardev socket,path=/tmp/qga.sock,server=on,wait=off,id=qga0 \
    # -device virtserialport,chardev=qga0,name=org.qemu.guest_agent.0 \
sleep 30
ssh -p 2201 -o "UserKnownHostsFile /dev/null" -o "StrictHostKeyChecking=no" -i ./output/xvm_key xdev@localhost 'bash -s' < ./bootstrap_magao-x.sh || exit 1
wait
echo "Finished installing MagAO-X software."