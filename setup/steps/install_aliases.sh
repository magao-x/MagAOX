#!/bin/bash
echo "alias teldump='logdump --dir=/data/telem --ext=.bintel'" | sudo tee /etc/profile.d/teldump.sh