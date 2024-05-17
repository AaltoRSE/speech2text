# speech2text

>*_NOTE:_* The non-technical user guide for the Open On Demand web interface can be found [here](https://aaltorse.github.io/speech2text/).

This repo contains instructions for setting up and applying the speech2text app on Aalto Triton cluster. The app utilizes

- [WhisperX](https://github.com/m-bain/whisperX) automatic speech recognition tool
- [wav2vec]() to find word start and end timestamps for WhisperX transcription
- [Pyannote](https://huggingface.co/pyannote/speaker-diarization) speaker detection (diarization) tool 

The speech recognition and diarization steps are run independently and their result segments are combined using a simple algorithm which for each transcription word segment finds the most overlapping (in time) speaker segment.

The required models are described [here](#models). 

Conda environment and Lmod setup is described [here](#setup). 

Command line (technical) usage on Triton is described [here](#usage).

Open On Demand web interface (non-technical) usage is described [here](https://aaltorse.github.io/speech2text/).

Supported languages are:

arabic (ar), armenian (hy), bulgarian (bg), catalan (ca), chinese (zh), czech (cs), danish (da), dutch (nl), english (en), estonian (et), finnish (fi), french (fr), galician (gl), german (de), greek (el), hebrew (he), hindi (hi), hungarian (hu), icelandic (is), indonesian (id), italian (it), japanese (ja), kazakh (kk), korean (ko), latvian (lv), lithuanian (lt), malay (ms), marathi (mr), nepali (ne), norwegian (no), persian (fa), polish (pl), portuguese (pt), romanian (ro), russian (ru), serbian (sr), slovak (sk), slovenian (sl), spanish (es), swedish (sv), thai (th), turkish (tr), ukrainian (uk), urdu (ur), vietnamese (vi)


## Deploy on Aalto Triton

Create a base folder for the app if haven't been created

```
mkdir /appl/manual_installations/software/speech2text
```

Clone git repo and change directory

```bash
cd /appl/manual_installations/software/speech2text
git clone https://github.com/AaltoRSE/speech2text.git YYYY-N
cd YYYY-N
```

where `YYYY` is current year and `N` the running version number for the year. 

Create a conda environment to `env/`

```bash
module load mamba
mamba env create --file env.yml --prefix env/
```

Run 

```bash
module load model-huggingface
bin/deploy
```

Check the contents of the script for details.


## Usage

### OnDemand

TODO: non-techincal user guide (video?)


### Command line

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


> **__NOTE:__** While speech2text by default uses the `large-v3` model, user can specify the model with the `SPEECH2TEXT_WHISPER_MODEL` environment variable. Note, however, that only `large-v2` and `large-v3` models have been pre-downloaded.


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


## Models

The required models have been downloaded beforehand from Hugging Face and saved into a shared data folder on the cluster. Therefore, users do not have to download the models themselves.
**Make sure the following models have been downloaded and accessible**.

### Faster Whisper  

We support `large-v2` and `large-v3` (default) multilingual [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) models. Languages supported by the models are:

The models are covered by the [MIT licence]((https://huggingface.co/models?license=license:mit)) and have been pre-downloaded from Hugging Face to 

`/scratch/shareddata/dldata/huggingface-hub-cache/hub/models--Systran--faster-whisper-large-v2`

and

`/scratch/shareddata/dldata/huggingface-hub-cache/hub/models--Systran--faster-whisper-large-v3`

### wav2vec

We use [wav2vec](https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) models as part of the diarization pipeline which efines the timestamps from whisper transcriptions using forced alignment a phoneme-based ASR model (wav2vec 2.0). This provides word-level timestamps, as well as improved segment timestamps.

We use a fine-tuned wav2vec model for each of the supported languages. All the models are fine-tuned over the [Meta's XLRS](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) model.

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

>**__IMPORTANT:__** This is out of date!

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

If a single audio file contains speech in more than one language, result files will (probably) still be produced but the results will (probably) be nonsensical to some extent. This is because WhisperX appears to translate languages to the specified target language (mandatory argument SPEECH2TEXT_LANGUAGE). Related discussion: [https://github.com/openai/whisper/discussions/49](https://github.com/openai/whisper/discussions/49).

In some cases, this problem can avoided relatively easily. For example, if the language changes only once in the middle of the audio, you can just split the file into two and process the parts separately.  You can use any audio processing software to do this, e.g. [Audacity](https://www.audacityteam.org/).

## Licensing

### Source Code

The source code in this repository is covered by the [MIT](./LICENSE.MIT) license.

### Audio Files

The example audio files `test-data/en/test1.mp3`, `test-data/en/test2.mp3`, and `test-data/fi/test1.mp3` in this repository, which are recordings of the repository owner's voice, are covered by the [CC-BY-NC-ND]((./LICENCE.CC-BY-NC-ND)) license.
