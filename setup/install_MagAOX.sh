#!/bin/bash
set -exuo pipefail
IFS=$'\n\t'
cd /opt/MagAOX/source/MagAOX
make setup
make all
make install