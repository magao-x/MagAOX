#!/usr/bin/env bash
ls -lah
mkdir -p ./output/bundle/
ls -R ./utm
cp -vR ./utm ./output/bundle/MagAO-X.utm
ls -R ./output/bundle/MagAO-X.utm
mv ./output/xvm.qcow2 ./output/bundle/MagAO-X.utm/Data/xvm.qcow2
cp ./output/xvm_key ./output/xvm_key.pub ./output/bundle/
cd ./output/bundle/
tar -cJvf ./MagAO-X_UTM.tar.xz ./*
