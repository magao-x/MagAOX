#!/usr/bin/env bash
ls -lah
mkdir -p ./output/bundle/MagAO-X.utm
cp -R ./utm ./output/bundle/MagAO-X.utm
mv ./output/xvm.qcow2 ./output/bundle/MagAO-X.utm/Data/xvm.qcow2
tar -cJvf ./MagAO-X_VM_bundle.tar.xz ./bundle
