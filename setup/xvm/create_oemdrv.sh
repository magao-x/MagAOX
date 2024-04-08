#!/usr/bin/env bash
mkdir -p input output
# make ssh key pair
if [[ ! -e ./output/xvm_key ]]; then
    ssh-keygen -q -t ed25519 -f ./output/xvm_key -N ''
fi
cp ./output/xvm_key.pub ./input/kickstart/authorized_keys
# copy kickstart files
cp -R ./kickstart ./input/
rm -f ./input/oemdrv.{dmg,qcow2}
if [[ $(uname) == "Darwin" ]]; then
    hdiutil create -srcfolder ./input/kickstart -format UDRO -volname "OEMDRV" -fs "MS-DOS FAT32" ./input/oemdrv.dmg
    qemu-img convert -f dmg -O qcow2 ./input/oemdrv.dmg ./input/oemdrv.qcow2
else
    qemu-img create -f raw oemdrv.img 1G
    sudo parted oemdrv.img --script -- mklabel msdos
    sudo parted oemdrv.img --script -- mkpart primary fat32 1MiB 100%
    sudo mkfs.fat -n OEMDRV -F 32 oemdrv.img
    sudo mkdir -p /mnt/oemdrv
    sudo mount -o loop oemdrv.img /mnt/oemdrv
    sudo cp -R ./input/kickstart/* /mnt/oemdrv/
    sudo umount /mnt/oemdrv
    qemu-img convert -f raw -O qcow2 oemdrv.img ./input/oemdrv.qcow2
fi
