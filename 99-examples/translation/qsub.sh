#!/bin/bash
#PBS -N translate
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=16gb:ngpus=1:gpu_cap=compute_60:gpu_mem=15gb
#PBS -l walltime=00:30:00
#PBS -m bae
# mail on begin, abort, end

echo ${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}

# TODO: ensure that the PYTHON variable is set to the correct Python interpreter (in the conda environment you have installed earlier)
export PYTHON="/storage/brno2/home/hrabalm/envs/npfl101demo/bin/python"
export SCRIPT_DIR="$PBS_O_WORKDIR"


cd "$SCRIPT_DIR"  # note that in some cases this can affect performance, depending on how the cwd is used
"$PYTHON" translate.py -i "test.en" -o "test.ja" -m "/storage/brno12-cerit/home/hrabalm/checkpoints/20241118_granite_sft_run8_completions_tied_dropout_decoupled/checkpoint-90000"

clean_scratch
