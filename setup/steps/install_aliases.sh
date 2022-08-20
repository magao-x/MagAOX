#!/bin/bash
echo "alias teldump='logdump --dir=/opt/MagAOX/telem --ext=.bintel'" | sudo tee /etc/profile.d/teldump.sh
