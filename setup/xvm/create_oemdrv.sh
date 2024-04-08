#!/usr/bin/env bash
rm -f ./input/oemdrv.{dmg,qcow2}
if [[ $(uname) == "Darwin" ]]; then
    hdiutil create -srcfolder ./input/kickstart -format UDRO -volname "OEMDRV" -fs "MS-DOS FAT32" ./input/oemdrv.dmg
    qemu-img convert -f dmg -O qcow2 ./input/oemdrv.dmg ./input/oemdrv.qcow2
else
    qemu-img create -f raw oemdrv.img 1G
    parted oemdrv.img --script -- mklabel msdos
    parted oemdrv.img --script -- mkpart primary fat32 1MiB 100%
    mkfs.fat -n OEMDRV -F 32 oemdrv.img
    mount -o loop oemdrv.img /mnt/oemdrv
    cp -R ./input/kickstart/* /mnt/oemdrv/
    umount /mnt/oemdrv
    qemu-img convert -f raw -O qcow2 oemdrv.img ./input/oemdrv.qcow2
fi
