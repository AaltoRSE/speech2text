help_text = [[

This app does speech2text with diarization.

Example run on a single file: 

    export SPEECH2TEXT_EMAIL=john.smith@aalto.fi
    export SPEECH2TEXT_LANGUAGE=finnish
    speech2text audiofile.mp3

Example run on a folder containing one or more audio file:

    export SPEECH2TEXT_EMAIL=jane.smith@aalto.fi
    export SPEECH2TEXT_LANGUAGE=finnish
    speech2text audiofiles/

Language must be provided from the list of supported languages:

arabic (ar), armenian (hy), bulgarian (bg), catalan (ca), chinese (zh), czech (cs), danish (da), 
dutch (nl), english (en), estonian (et), finnish (fi), french (fr), galician (gl), german (de), 
greek (el), hebrew (he), hindi (hi), hungarian (hu), icelandic (is), indonesian (id), 
italian (it), japanese (ja), kazakh (kk), korean (ko), latvian (lv), lithuanian (lt), malay (ms), 
marathi (mr), nepali (ne), norwegian (no), persian (fa), polish (pl), portuguese (pt), 
romanian (ro), russian (ru), serbian (sr), slovak (sk), slovenian (sl), spanish (es), 
swedish (sv), thai (th), turkish (tr), ukrainian (uk), urdu (ur), vietnamese (vi)

The audio files can be in any common audio (.wav, .mp3, .aff, etc.) or video (.mp4, .mov, etc.) format.

The speech2text app writes result files to a subfolder results/ next to each audio file.
Result filenames are the audio filename with .txt and .csv extensions. For example, result files
corresponding to audiofile.mp3 are written to results/audiofile.txt and results/audiofile.csv.
Result files in a folder audiofiles/ will be written to folder audiofiles/results/.

Notification emails will be sent to SPEECH2TEXT_EMAIL. If SPEECH2TEXT_EMAIL is left 
unspecified, no notifications are sent.
]]

local version = "20240408"
whatis("Name : Aalto speech2text")
whatis("Version :" .. version)
help(help_text)

local speech2text = "/share/apps/manual_installations/speech2text/" .. version .. "/bin/"
local conda_env = "/share/apps/manual_installations/speech2text/" .. version .. "/env/bin/"

prepend_path("PATH", speech2text)
prepend_path("PATH", conda_env)

local hf_home = "/scratch/shareddata/dldata/huggingface-hub-cache/"
local pyannote_cache = hf_home .. "hub/"
local torch_home = "/scratch/shareddata/speech2text"
local pyannote_config = "/share/apps/manual_installations/speech2text/" .. version .. "/pyannote/config.yml"
local numba_cache = "/tmp" 
local mplconfigdir = "/tmp"

pushenv("HF_HOME", hf_home)
pushenv("PYANNOTE_CACHE", pyannote_cache)
pushenv("TORCH_HOME", torch_home)
pushenv("XDG_CACHE_HOME", torch_home)
pushenv("PYANNOTE_CONFIG", pyannote_config)
pushenv("NUMBA_CACHE_DIR", numba_cache)
pushenv("MPLCONFIGDIR", mplconfigdir)

local speech2text_mem = "8G"
local speech2text_cpus_per_task = "6"
local speech2text_tmp = os.getenv("WRKDIR") .. "/.speech2text"

pushenv("SPEECH2TEXT_MEM", speech2text_mem)
pushenv("SPEECH2TEXT_CPUS_PER_TASK", speech2text_cpus_per_task)
pushenv("SPEECH2TEXT_TMP", speech2text_tmp)

pushenv("HF_HUB_OFFLINE", "1")

if mode() == "load" then
    LmodMessage("For more information, run 'module spider speech2text/" .. version .. "'")
end
 
