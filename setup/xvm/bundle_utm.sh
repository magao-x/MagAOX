#!/usr/bin/env bash
cp -R ./utm ./output/MagAO-X.utm
cd ./output
mv ./xvm.qcow2 ./MagAO-X.utm/Data/xvm.qcow2
tar -cJvf ./MagAO-X_VM_bundle.tar.xz ./MagAO-X.utm
