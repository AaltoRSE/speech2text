# speech2text

This repo contains instructions for setting up and applying the `speech2text` app on Aalto Triton cluster. The app utilizes [OpenAI Whisper](https://openai.com/research/whisper) automatic speech recognition and [Pyannote speaker detection (diarization)](https://huggingface.co/pyannote/speaker-diarization) tools. The speech recognition and diarization steps are run sequentially (and independently) and their results are combined (aligned) using a script from [https://github.com/yinruiqing/pyannote-whisper.git](https://github.com/yinruiqing/pyannote-whisper.git). 

The recommended way to setup and run `speech2text` is to use a [Conda environment](#setup-and-run-speech2text-in-a-conda-environment). In addition, the repo provides instructions to to load and run the app using a clean user interface via an [Lmod module](#setup-and-run-speech2text-with-lmod).

## Models

The required Whisper, Huggingface, and Pyannote models are downloaded beforehand and saved into a shared data folder on the cluster. Therefore, users do not have to download the models themselves. 

### OpenAI Whisper  

We use `large`, the biggest and most accurate multilingual Whisper model. Languages supported by the model are described [here](https://github.com/openai/whisper#available-models-and-languages). The model has been pre-downloaded to `/scratch/shareddata/openai-whisper/envs/venv-2023-03-29/models/large-v2.pt`. 

> **_NOTE:_**  To use the English-specific model, use the `--english` option. However, the largest English-specific model `medium.en` [is reported to be comparable in accuracy]((https://github.com/openai/whisper#available-models-and-languages)) to the multilingual `large` model for English. The model has been pre-downloaded to `/scratch/shareddata/openai-whisper/envs/venv-2023-03-29/models/medium.en`.
 
[LICENCE: MIT](https://github.com/openai/whisper/blob/main/LICENSE)

### HuggingFace speechbrain/spkrec-ecapa-voxceleb

The [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) embeddings have
been pre-downloaded to `/scratch/shareddata/speech2text/huggingface/hub/models--speechbrain--spkrec-ecapa-voxcelebs`.

[LICENCE: MIT](https://huggingface.co/models?license=license:mit)

### Pyannote 

The [pyannote/segmentation](https://huggingface.co/pyannote/segmentation/blob/main/pytorch_model.bin) model has
been pre-downloaded to `/scratch/shareddata/speech2text/pyannote/segmentation/blob/main/pytorch_model.bin`.

This path needs to be added to [config file](https://huggingface.co/pyannote/speaker-diarization/blob/main/config.yaml) `pyannote/config.yml`. (Has been added as a default.)

[LICENCE: MIT](https://huggingface.co/models?license=license:mit)

> **_NOTE:_**  
> The model is free to use but gated: In order to use it, go to [pyannote/segmentation](https://huggingface.co/pyannote/segmentation/blob/main/pytorch_model.bin), log in as Hugging Face user, and accept the conditions to access it.

## Data

Repository contains two example audio files for testing purposes
```
test-data/test_en.mp3
test-data/test_fi.mp3
```

## Setup and run speech2text in a Conda environment

Clone git repo and change directory
```bash
git clone https://github.com/AaltoRSE/speech2text.git
cd speech2text
```

Create a conda environment to `./env/`
```bash
module load miniconda
mamba env create --file env.yml --prefix env/
```

Set model location environment variables `HF_HOME` (Hugging Face models) and `WHISPER_CACHE` (OpenAI Whisper models):
```bash
export HF_HOME="/scratch/shareddata/speech2text"
export WHISPER_CACHE="/scratch/shareddata/openai-whisper/envs/venv-2023-03-29/models/"
```

Make sure the path to Pyannote segmentation model in `pyannote/config.yml:8` is valid.

Run using an interactive Slurm job:
```bash
srun --time=12:00:00 \
     --mem=12G \
     --cpus-per-task 8 \
     env/bin/python3 src/speech2text.py test-data/
```

The `speech2text.py` app loops over all audio files in the target folder (`test-data/`) and writes result files to a `results/` subfolder (`test-data/results/`). Output filenames for each input file are the input filename with `.txt` and `.csv` extensions. For example, the result files corresponding to `test-data/audiofile.wav` are `test-data/results/audiofile.txt` and `test-data/results/audiofile.csv`. The default output folder can be changed using the `--output-dir` option, for example
```bash
srun --time=12:00:00 \
     --mem=12G \
     --cpus-per-task 8 \
     env/bin/python3 src/speech2text.py data/ --output-dir my-results/
```

> **_NOTE:_** The audio files can contain speech in any [supported language](https://github.com/openai/whisper#available-models-and-languages). If the input audio files contain speech in other than one of the supported languages, results will still be produced but are most probably nonsense.

> **_NOTE:_** If one or more of the input files are not of `.wav` format, they will be converted to `.wav` files within the script. The resulting `.wav` files are by default removed after the script has been executed, but can be retained by using option `--keep-converted-files`. The path to each converted file is the same as the original file with `.wav` extension: for example, `test-data/audiofile.mp3` will be converted to `test-data/audiofile.wav`. The original audio files are always retained.


## Setup and run speech2text with Lmod

The module is designed to hide all Slurm-related commands to make running the script less daunting for people with no technical background.

The required commands are as follows. Activate and load the module with
```bash
module use /share/apps/manual_installations/speech2text/modules
module load speech2text
```
Run `speech2text.py` with 
```bash
speech2text test-data/
```
Add options for `speech2text` as you would for `speech2text.py`, for example
```bash
speech2text test-data/ --output-dir my-results/ --keep-converted-files
```

Check `modules/speech2text/20230720.lua` and `bin/speech2text` for details of the module implementation.



## Output formats

The output formats are `.csv` and `.txt`. For example, output files corresponding to input audio files
```bash
test_en.mp3
test_fi.mp3
```
are
```bash
test_en.csv
test_en.txt
test_fi.csv
test_fi.txt
```

Example of `.csv` output format:
```
start,end,start_str,end_str,speaker,transcription
0,10,00:00:00,00:00:10,SPEAKER_00,"Tämä on esimerkki ensimmäiseltä puhujalta."
11,15,00:00:11,00:00:15,SPEAKER_01,"This is an example from a second speaker."
```

Corresponding example of `.txt` output format:
```
(00:00:00 - 00:00:10) SPEAKER_00

Tämä on esimerkki ensimmäiseltä puhujalta.

(00:00:11 - 00:00:15) SPEAKER_01

This is an example from a second speaker.
```

## Development

To lint/autoformat, run
```bash
black src && isort src
```

## Build and run with Singularity

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

## Known Issues

### Inference using CPUs versus GPUs

The recommended way to do inference with Whisper is to use GPUs. However, on Triton, we have to make a compromise between GPU queues and inference efficiency. All the scripts use CPUs by default.

### Increasing the number of CPUs for inference

There is a plateauing problem with running Whisper inference with multiple CPUs (not GPUs). Increasing the number of CPUs speeds up inference until around 8 CPUs but plateaus and begins to slow down after 16. See related discussion where same behavior has been observed: [https://github.com/ggerganov/whisper.cpp/issues/200](https://github.com/ggerganov/whisper.cpp/issues/200) Therefore, in all the scripts, the number of CPUs is set to 8 by default.

## Licensing

### Source Code

The source code in this repository is covered by the MIT license. See [LICENSE](./LICENSE) for details.

### Audio Files

The example audio files `test-data/test_en.mp3` and `test-data/test_fi.mp3` in this repository, which are recordings of the repository owner's voice, are also covered by the MIT license. You are free to use, modify, and distribute this file under the terms of the MIT license as provided in the [LICENSE](./LICENSE) file.
