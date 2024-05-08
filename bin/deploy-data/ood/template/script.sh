#!/bin/bash -l
export PATH="/appl/manual_installations/software/speech2text/<VERSION>/env/bin"

export HF_HOME="<HF_HOME>"
export PYANNOTE_CACHE="<PYANNOTE_CACHE>"
export TORCH_HOME="<TORCH_HOME>"
export XDG_CACHE_HOME="<TORCH_HOME>"
export PYANNOTE_CONFIG="<PYANNOTE_CONFIG>"
export NUMBA_CACHE_DIR="<NUMBA_CACHE_DIR>"
export MPLCONFIGDIR="<MPLCONFIGDIR>"

export SPEECH2TEXT_MEM="<SPEECH2TEXT_MEM>"
export SPEECH2TEXT_CPUS_PER_TASK="<SPEECH2TEXT_CPUS_PER_TASK>"

export HF_HUB_OFFLINE="1"

module load cuda
speech2text $audio_path
