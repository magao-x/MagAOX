#!/usr/bin/env bash
echo "Starting up the VM for MagAO-X dependencies installation..."
source ./_common.sh
if [[ -e ./output/xvm_stage1.qcow2 ]]; then
    cp ./output/xvm_stage1.qcow2 ./output/xvm.qcow2
elif [[ ! -e ./output/xvm.qcow2 ]]; then
    echo "No xvm.qcow2 found for stage 2"
    exit 1
fi

qemu-system-${vmArch} \
    -name xvm \
    -netdev user,id=user.0,hostfwd=tcp:127.0.0.1:2201-:22 \
    -device virtio-keyboard-pci -device virtio-mouse-pci \
    -smp 3 \
    $qemuMachineFlags \
    -drive if=pflash,format=raw,id=ovmf_code,readonly=on,file=./output/firmware_code.fd \
    -drive if=pflash,format=raw,id=ovmf_vars,file=./output/firmware_vars.fd \
    -drive file=output/xvm.qcow2,format=qcow2 \
    -device virtio-gpu-pci \
    -device virtio-net-pci,netdev=user.0 \
    -m 8192M \
    $ioFlag \
&
sleep 60
ssh -p 2201 -o "UserKnownHostsFile /dev/null" -o "StrictHostKeyChecking=no" -i ./output/xvm_key xdev@localhost mkdir -p MagAOX
rsync -rv --exclude xvm -e 'ssh -p 2201 -o "UserKnownHostsFile /dev/null" -o "StrictHostKeyChecking=no" -i ./output/xvm_key' ../../ xdev@localhost:MagAOX/
ssh -p 2201 -o "UserKnownHostsFile /dev/null" -o "StrictHostKeyChecking=no" -i ./output/xvm_key xdev@localhost 'bash -s' < ./bootstrap_magao-x.sh || exit 1
# wait for the backgrounded qemu process to exit:
wait
mv -v ./output/xvm.qcow2 ./output/xvm_stage2.qcow2
echo "Finished installing MagAO-X dependencies."
