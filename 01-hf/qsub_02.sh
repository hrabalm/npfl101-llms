#!/bin/bash
#PBS -N task02
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=16gb:ngpus=1:gpu_cap=compute_70:gpu_mem=15gb
#PBS -l walltime=00:30:00
#PBS -m bae
# mail on begin, abort, end

source "$PBS_O_WORKDIR/common.sh"

FILE="$SCRIPT_DIR/02_train.py"
cd "$SCRIPT_DIR"  # note that in some cases this can affect performance, depending on how the cwd is used
"$PYTHON" "$FILE"

clean_scratch
