#!/usr/bin/env bash
source ./_common.sh
mkdir -p input/kickstart output
# generate kickstart template
cat ./kickstart/ks.cfg.template | envsubst > ./input/kickstart/ks.cfg
cp ./output/xvm_key.pub ./input/kickstart/authorized_keys
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
