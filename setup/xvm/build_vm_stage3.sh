#!/usr/bin/env bash
echo "Starting up the VM for MagAO-X dependencies installation..."
source ./_common.sh
if [[ -e ./output/xvm_stage2.qcow2 ]]; then
    cp ./output/xvm_stage2.qcow2 ./output/xvm.qcow2
elif [[ ! -e ./output/xvm.qcow2 ]]; then
    echo "No existing xvm.qcow2 found to use in stage 3"
    exit 1
fi
$qemuSystemCommand &
sleep 60
ssh -p 2201 -o "UserKnownHostsFile /dev/null" -o "StrictHostKeyChecking=no" -i ./output/xvm_key xdev@localhost mkdir -p MagAOX
updateGuestMagAOXCheckout  # since the previous stage VM may be from cache
ssh -p 2201 -o "UserKnownHostsFile /dev/null" -o "StrictHostKeyChecking=no" -i ./output/xvm_key xdev@localhost 'bash -s' < ./guest_provision_up_to_build.sh || exit 1
# wait for the backgrounded qemu process to exit:
wait
mv -v ./output/xvm.qcow2 ./output/xvm_stage3.qcow2
echo "Finished installing MagAO-X dependencies."
