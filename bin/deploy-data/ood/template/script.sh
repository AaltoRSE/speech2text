#!/bin/bash -l
export PATH="/share/apps/manual_installations/speech2text/<VERSION>/env/bin:/usr/lib64/ccache:/usr/local/bin:/usr/bin:/home/firoozh1/bin:/usr/local/sbin:/usr/sbin:/opt/ibutils/bin:/opt/dell/srvadmin/bin"

export HF_HOME="<HF_HOME>"
export PYANNOTE_CACHE="<PYANNOTE_CACHE>"
export TORCH_HOME="<TORCH_HOME>"
export XDG_CACHE_HOME="<XDG_CACHE_HOME>"
export PYANNOTE_CONFIG="<PYANNOTE_CONFIG>"
export NUMBA_CACHE_DIR="<NUMBA_CACHE_DIR>"
export MPLCONFIGDIR="<MPLCONFIGDIR>"

export SPEECH2TEXT_MEM="<SPEECH2TEXT_MEM>"
export SPEECH2TEXT_CPUS_PER_TASK="<SPEECH2TEXT_CPUS_PER_TASK>"
export SPEECH2TEXT_TMP="<SPEECH2TEXT_TMP>"

export HF_HUB_OFFLINE="1"

speech2text $folder_path
