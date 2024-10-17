#!/bin/bash
#PBS -N task01
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=16gb:ngpus=1:gpu_cap=compute_70:gpu_mem=15gb
#PBS -l walltime=00:30:00
#PBS -m bae
# mail on begin, abort, end

source "$SCRIPT_DIR/common.sh"

FILE="$SCRIPT_DIR/01_create_dataset.py"
"$PYTHON" "$FILE"

clean_scratch
