#!/bin/bash
#PBS -N task01
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=16gb:ngpus=1:gpu_cap=compute_70:gpu_mem=15gb
#PBS -l walltime=00:30:00
#PBS -m bae  # mail on begin, abort, end

echo ${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}

SECRETS_FILE="$PBS_SCRIPT_DIR/secrets.sh"
if [ -e "$SECRETS_FILE" ]
then
    source "$SECRETS_FILE"
else
    echo "Secrets file not found: $SECRETS_FILE"
fi

source "$PBS_SCRIPT_DIR/common.sh"

FILE="$PBS_SCRIPT_DIR/01_create_dataset.py"
"$PYTHON" "$FILE"

clean_scratch
