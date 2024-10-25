# 01-hf

## Changelog

### 20241025

I tried to make the instructions a bit more clear and also cover a few key points I forgot to mention during the lab. I also install a few more packages (e.g. `flash-attn`) which can help some models use more efficient implementation of attention mechanism to save some computational time and decrease VRAM usage.

If you want to install the updated environment, the easiest way is to remove the whole prefix (e.g. `rm -rf ~/envs/npfl101demo`) and install it again, because sometimes packages don't support the newest versions of other packages. A common example would be the newest version of PyTorch which is often not supported for a while until developers catch up.

You should also take a look at the new information about accessing the gated models. I have forgotten to mention this during the lab.

## Introduction

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
```

## Install

First, check what is your current home directory

```bash
pwd
```

You should see something like this:

```raw
/storage/brno12-cerit/home/USERNAME
```

```raw
/storage/brno2-cerit/home/USERNAME
```

You should take a note of this directory, because it is where we will be installing Miniforge3 and Conda environment.

If you are running this inside an interactive job (see above), you should also export the `TMPDIR` variable so that it points to the scratch folder. This means that the temporary files created during installation will use the fast local storage you asked for instead of `/tmp` directory that has relatively low usage limit (often ~1GB).

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
~/miniforge3/bin/mamba create --prefix=~/envs/npfl101demo python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 ipython cuda-python cuda-nvcc=12.1 xformers::xformers flash-attn -c pytorch -c nvidia
```

Install pip packages

```bash
~/envs/npfl101demo/bin/pip install --no-cache-dir transformers datasets wandb pandas trl peft bitsandbytes
```

If running in an interactive job, you should also always clean the scratch.

```bash
clean_scratch
```

You can now close the interactive job and continue using your frontend.

## Exercises

Firstly, look into `qsub_01.sh` and try to understand what it does.
At the top, it configures the OpenPBS job which will be run on the cluster and
specifies resources required, you will have to modify these to fit your
requirements later, for details check Metacentrum documentation.

Note that jobs requering GPUs have to be submitted to gpu queue (or another GPU
queue you have access to). See the difference between `qsub_01.sh` and `qsub_02.sh`.

Metacentrum also has an interactive QSub Planner which can be useful to check
what nodes satisfy the given requirements (if any).

Before we continue, you have to edit `common.sh` file and update the `PYTHON`
variable so that it points to the Python executable in the conda environment we
have installed earlier.

Also, if you want to use the gated models on Huggingface (meaning that you have to have an Huggingface account and explicitly agree with some conditions to access some models) you have to generate a access token on [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). You then have to somehow share the token with the libraries that need them. There are several ways of doing this, here we will set an `HF_TOKEN` environment variable.

To do this, create a copy of the `secrets.example.sh` file:

```bash
cp secrets.example.sh secrets.sh
```

Afterwards, fill in your `HF_TOKEN`. Note, that these tokens should generally not be commited to your GitHub repository, because you can later decide to publish even private repositories and forget about them (and properly purging them from the history can be tricky - in that case you should generally revoke old tokens and generate new ones anyway.)

Now you can run prepared scripts using `qsub`. First look into a given Python script, go through them and check whether there are any TODOs left. Then you can add them to the queue.

```bash
qsub qsub_01.sh
```

After the job finished, it will create two files in the directory you were enqueueing them from, which capture the standard and error outputs of the job. For checking the progress while it is running, check the Tips section and [relevant docs](https://docs.metacentrum.cz/computing/job-tracking/).

Once again, you should go through (or at least skim through) the offical documentation, because there's much I was not able to cover here.

Also, in each of the Python scripts, I propose some optional exercises and things you might want to look into, depending on what can be useful for your project.

## Tips

You can check the state of your jobs with `qstat`

```bash
qstat -u USERNAME
```

Look at the job details

```bash
qstat -f JOB_ID
```

Check job progress/outputs while it is running

```bash
qstat -u USERNAME # find job id
qstat -f JOB_ID | grep host  # find host
cd /var/spool/pbs/spool
tail -f JOB_ID*  # e.g. tail -f 123456*
```
