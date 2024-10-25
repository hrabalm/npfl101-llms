echo ${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}

SECRETS_FILE="$PBS_O_WORKDIR/secrets.sh"
if [ -e "$SECRETS_FILE" ]
then
    source "$SECRETS_FILE"
else
    echo "Secrets file not found: $SECRETS_FILE"
fi

export PYTHON="/storage/brno2/home/hrabalm/envs/npfl101demo/bin/python"
export SCRIPT_DIR="$PBS_O_WORKDIR"
