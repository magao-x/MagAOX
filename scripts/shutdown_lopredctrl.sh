#!/bin/bash

# Script to destroy a cacao shmim
# Usage: ./destroy_shmim.sh <name>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <shmim_name>"
    exit 1
fi

NAME=${1:-lopredctrl}

cacao << EOF
rmshmim $NAME
quit
EOF

echo "Destroyed shmim $NAME"