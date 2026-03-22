#!/bin/bash

# Script to create a cacao shmim with size Nx1
# Usage: ./setup_lopredctrl.sh [name] [N]

NAME=${1:-lopredctrl}
N=${2:-100}

cacao << EOF
mk2Dim "s>tf32>$NAME" $N 1
quit
EOF

echo "Created shmim $NAME with size ${N}x1"