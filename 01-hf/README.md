# 01-hf

For these examples, please use the zenith frontend which is located in Brno and uses brno12-cerit storage.

```bash
ssh USERNAME@zenith.metacentrum.cz
```

As a small note, Metacentrum uses Kerberos for authentication, so using just `ssh-copy-id` unfortunately does not work. See [here](https://docs.metacentrum.cz/access/kerberos/) for instructions.

## Interactive jobs

In case you are or some packages are compiled from source, installation can take a long time and need more resources such as CPUs, RAM or scratch storage for temporary files.

In that case you can ask for an interactive task:

```bash
qsub -l select=1:ncpus=1:mem=8gb:scratch_local=16gb -I -l walltime=1:00:00
```

In this case, you should also specify the cluster or the city explicitly so that the remote storage you manipulate is part of the same cluster.

```bash
qsub -l select=1:ncpus=1:mem=8gb:scratch_local=16gb:brno=True -I -l walltime=1:00:00
# or
qsub -l select=1:ncpus=1:mem=8gb:scratch_local=16gb:cl_adan=True -I -l walltime=1:00:00
```

## Install

First, check what is your current home directory

```bash
pwd
```

You should see something like this:

```raw
/storage/brno12-cerit/home/hrabalm
```

If you are running this inside an interactive job (see above), you should also export set the `TMPDIR` variable so that it points to the scratch folder. This means that the temporary files created during installation will use the fast local storage you asked for instead of `/tmp` directory that has relatively low usage limit.

```bash
# only when running inside interactive job!
export TMPDIR="$SCRATCHDIR"
```

Install miniforge3 (this includes mamba - a faster reimplementation of conda) to the current node's home directory (or different place depending on your preferences)

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Create a conda prefix with Python 3.11 and PyTorch

```bash
~/miniforge3/bin/mamba create --prefix=~/envs/npfl101demo python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 ipython -c pytorch -c nvidia
```

Install pip packages

```bash
~/envs/npfl101demo/bin/pip install --no-cache-dir transformers datasets wandb pandas trl peft bitsandbytes
```

If running in an interactive job, you should also always clean the scratch.

```bash
clean_scratch
```

## Exercises

Now you can run prepared scripts using `qsub`. For example, you can run the first task like this:

```bash
qsub qsub_01.sh
```

## Tips

You can check the state of your jobs with `qstat`

```bash
qstat -u USERNAME
```

Look at the job details

```bash
qstat -f JOB_ID
```

Check job progress/output

```bash
qstat -u USERNAME # find job id
qstat -f JOB_ID | grep host  # find host
cd /var/spool/pbs/spool
tail -f JOB_ID*  # e.g. tail -f 123456*
```
