# speech2text

This repo contains instructions for setting up and applying the speech2text app on Aalto Triton cluster. The app utilizes [WhisperX](https://github.com/m-bain/whisperX) automatic speech recognition tool and [Pyannote](https://huggingface.co/pyannote/speaker-diarization) speaker detection (diarization) pipeline. The speech recognition and diarization steps are run independently and their result segments are combined (aligned) using a simple algorithm which for each transcription segment finds the most overlapping (in time) speaker segment.

The required models are described [here](#models). 

Conda environment and Lmod setup is described [here](#setup). 

Usage is describe [here](#usage).

The non-technical user guide using the Open On Demand web interface can be found [here](https://aaltorse.github.io/speech2text/).

## Models

The required models have been downloaded beforehand from Hugging Face and saved into a shared data folder on the cluster. Therefore, users do not have to download the models themselves.
**Make sure the following models have been downloaded and accessible**.

### Faster Whisper  

We use `large-v3`, currently the biggest and most accurate multilingual [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) model available. Languages supported by the model are:

afrikaans, arabic, armenian, azerbaijani, belarusian, bosnian, bulgarian, catalan, 
chinese, croatian, czech, danish, dutch, english, estonian, finnish, french, galician, 
german, greek, hebrew, hindi, hungarian, icelandic, indonesian, italian, japanese, 
kannada, kazakh, korean, latvian, lithuanian, macedonian, malay, marathi, maori, nepali,
norwegian, persian, polish, portuguese, romanian, russian, serbian, slovak, slovenian, 
spanish, swahili, swedish, tagalog, tamil, thai, turkish, ukrainian, urdu, vietnamese, 
welsh

The model is covered by the [MIT licence]((https://huggingface.co/models?license=license:mit)) and has been pre-downloaded from Hugging Face to 

`/scratch/shareddata/dldata/huggingface-hub-cache/hub/models--Systran--faster-whisper-large-v3`
 
### Pyannote 

The diarization is performed using the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) pipeline installed via [`pyannote.audio`](https://github.com/pyannote/pyannote-audio).
> **_NOTE:_**
> pyannote.audio is covered by [MIT licence]((https://huggingface.co/models?license=license:mit)) but the diarization pipeline is gated. In order to use it, log in to [Hugging Face](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the conditions to access it.

The [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation/blob/main/pytorch_model.bin) model used by the pipeline has been pre-downloaded from Hugging Face to 

`/scratch/shareddata/speech2text/pyannote/segmentation-3.0/blob/main/pytorch_model.bin`

This path has been hard-coded to the [Pyannote config file](https://huggingface.co/pyannote/speaker-diarization-3.1/blob/main/config.yaml) located in [pyannote/config.yml](pyannote/config.yml).


> **_NOTE:_**
> pyannote/segmentation-3.0 is also covered by [MIT licence]((https://huggingface.co/models?license=license:mit)) but is gated separately. In order to use it, log in to [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept the conditions to access it.
>
> Due to gating, the model has **not** been saved to `/scratch/shareddata/dldata/huggingface-hub-cache/` which is meant for models accessible more generally to Triton users.

Wrapper around the [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) pretrained speaker embedding model is used by pyannote-audio version 3.1 and higher (see [pyannote/config.yml](pyannote/config.yml)). The model is covered by the [MIT licence]((https://huggingface.co/models?license=license:mit)) and has been pre-downloaded from Hugging Face to 

`/scratch/shareddata/dldata/huggingface-hub-cache/hub/models--pyannote--wespeaker-voxceleb-resnet34-LM`.


## Data

Repository contains three example audio files for testing purposes
```
test-data/en/test1.mp3
test-data/en/test2.mp3
test-data/fi/test1.mp3
```

## Setup 

How to setup speech2text on Aalto Triton cluster.

### Conda environment

Create a base folder for the app and change directory

```
mkdir /share/apps/manual_installations/speech2text
cd /share/apps/manual_installations/speech2text
```

Clone git repo and change directory

```bash
git clone https://github.com/AaltoRSE/speech2text.git 00000000
cd 00000000
```

where `00000000` is the version number ("date") for development. 

Create a conda environment to `env/`

```bash
module load miniconda
mamba env create --file env.yml --prefix env/
```

Finally, make sure the path to [Pyannote segmentation model](#pyannote) in `pyannote/config.yml:8` is valid.

### Lmod

Activate the speech2text module with

```bash
module use /share/apps/manual_installations/speech2text/modules/speech2text/00000000.lua
```

Alternatively, copy the .lua script to `/share/apps/modules/speech2text/` so that it gets activated automatically at login.

Essentially, the module implementation prepends `/share/apps/manual_installations/speech2text/bin` and `/share/apps/manual_installations/speech2text/env/bin` to `PATH` and sets good default values for Slurm job resource requests for easy load and usage. Check [modules/speech2text/00000000.lua](modules/speech2text/00000000.lua) and [bin/speech2text](bin/speech2text) for details. 


## Usage

After the conda environment and Lmod have been [setup and activated](#setup), speech2text is used in three steps:

Load the speech2text app with

```bash
module load speech2text
```

Set email (for Slurm job notifications) and audio language environment variables with

```
export SPEECH2TEXT_EMAIL=my.name@aalto.fi
export SPEECH2TEXT_LANGUAGE=mylanguage
```

For example:

```
export SPEECH2TEXT_EMAIL=john.smith@aalto.fi
export SPEECH2TEXT_LANGUAGE=finnish
```

The following variables are already set by the Lmod .lua script. They can be ignored by user.

```
HF_HOME
TORCH_HOME
PYANNOTE_CONFIG
NUMBA_CACHE
MPLCONFIGDIR
SPEECH2TEXT_TMP
SPEECH2TEXT_MEM
SPEECH2TEXT_CPUS_PER_TASK
```

Note that you can leave the language variable unspecified, in which case speech2text tries to detect the language automatically. Specifying the language explicitly is, however, recommended.

Notification emails will be sent to given email address. If the addresss is left unspecified,
no notifications are sent.

Finally, process a single audio file with

```bash
speech2text test-data/en/test1.mp3
```

Alternatively, process multiple audio files in a folder

```bash
speech2text audio-files/
```

Using the latter option submits the files as an [array job](https://scicomp.aalto.fi/triton/tut/array/). See [src/submit.py](src/submit.py) for details about the submissions.

The audio file(s) can be in any common audio (.wav, .mp3, .aff, etc.) or video (.mp4, .mov, etc.) format. 

The transcription and diarization results (.txt and .csv files) corresponding to each audio file will be written to `results/` next to the file. See [below](#output-formats) for details.


## Output formats

The output formats are `.csv` and `.txt`. For example, output files corresponding to input audio files
```bash
test1.mp3
test2.mp3
```
are
```bash
test1.csv
test1.txt
test2.csv
test2.txt
```

Example of `.csv` output format (computer-friendly format):
```
start,end,speaker,transcription
00:00:00,00:00:05,SPEAKER_00,"This is the first sentence of the first speaker."
00:00:06,00:00:10,SPEAKER_00,"This is the second sentence of the first speaker."
00:00:11,00:00:15,SPEAKER_01,"This is a sentence from the second speaker."
```

Corresponding example of `.txt` output format (human-friendly format):
```
(00:00:00 - 00:00:10) SPEAKER_00

This is the first sentence of the first speaker. This is the second sentence of the first speaker.

(00:00:11 - 00:00:15) SPEAKER_01

This is a sentence from the second speaker.
```

## Tests and linting

Create development environment
```bash
mamba create --file env_dev.yml --prefix env-dev/
mamba activate env-dev/
```

Run unit tests in `src/`
```bash
pytest src
```

Lint code in `src/`
```
black src && isort src
```

## Build and run with Singularity

>**__NOTE:__** This maybe out of date!

Although currently not needed, the repo also contains a Singularity definition file `speech2text.def` in project root. 

### Build Singularity image

Clone git repo and change directory
```bash
git clone https://github.com/AaltoRSE/speech2text.git
cd speech2text
```

Build a singularity image file (`.sif`) with
```
srun --mem=8G --cpus-per-task 2 --time 1:00:00 singularity build speech2text.sif speech2text.def
```

### Run in Singularity container

Example run:
```bash
srun cpu --mem=10G --cpus-per-task 8 --time 12:00:00 singularity run --nv --bind /scratch:/scratch speech2text.sif test-data/
```

## Build documentation locally

To build documentation locally using Sphinx, run
```
python3 -m pip install -r docs/sphinx_requirements.txt
sphinx-build docs/source docs/build
```
The documentation can be found in `docs/build/`. A good place to start is the index page `docs/build/index.html`.


## Known Issues

### Audio files with more than one language

If a single audio file contains speech in more than one language, result files will (probably) still be produced but the results will (probably) be nonsensical to some extent. This is because even when using automatic language detection, Whisper appears to [detect the first language it encounters (if not given specifically) and stick to it until the end of the audio file, translating other encountered languages to the first language](https://github.com/openai/whisper/discussions/49).

In some cases, this problem is easily avoided. For example, if the language changes only once in the middle of the audio, you can just split the file into two and process the parts separately.  You can use any audio processing software to do this, e.g. [Audacity](https://www.audacityteam.org/).

## Licensing

### Source Code

The source code in this repository is covered by the [MIT](./LICENSE.MIT) license.

### Audio Files

The example audio files `test-data/en/test1.mp3`, `test-data/en/test2.mp3`, and `test-data/fi/test1.mp3` in this repository, which are recordings of the repository owner's voice, are covered by the [CC-BY-NC-ND]((./LICENCE.CC-BY-NC-ND)) license.
