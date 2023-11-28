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

The audio files can be in any common audio (.wav, .mp3, .aff, etc.) or video (.mp4, .mov, etc.) format.

The speech2text app writes result files to a subfolder results/ next to each audio file.
Result filenames are the audio filename with .txt and .csv extensions. For example, result files
corresponding to audiofile.mp3 are written to results/audiofile.txt and results/audiofile.csv.
Result files in a folder audiofiles/ will be written to folder audiofiles/results/.

Notification emails will be sent to SPEECH2TEXT_EMAIL. If SPEECH2TEXT_EMAIL is left 
unspecified, no notifications are sent.

Supported languages are:

afrikaans, arabic, armenian, azerbaijani, belarusian, bosnian, bulgarian, catalan, 
chinese, croatian, czech, danish, dutch, english, estonian, finnish, french, galician, 
german, greek, hebrew, hindi, hungarian, icelandic, indonesian, italian, japanese, 
kannada, kazakh, korean, latvian, lithuanian, macedonian, malay, marathi, maori, nepali,
norwegian, persian, polish, portuguese, romanian, russian, serbian, slovak, slovenian, 
spanish, swahili, swedish, tagalog, tamil, thai, turkish, ukrainian, urdu, vietnamese, 
welsh

You can leave the language variable SPEECH2TEXT_LANGUAGE unspecified, in which case 
speech2text tries to detect the language automatically. Specifying the language 
explicitly is, however, recommended.

Troubleshooting:

Requested Slurm resources can be adjusted using the following environment variables (default in parentheses):
- SPEECH2TEXT_MEM ("12G")
- SPEECH2TEXT_CPUS_PER_TASK (6)
- SPEECH2TEXT_TIME ("24:00:00")

]]

local version = "00000000"
whatis("Name : Aalto speech2text")
whatis("Version :" .. version)
help(help_text)

local speech2text = "/share/apps/manual_installations/speech2text/" .. version .. "/bin/"
local conda_env = "/share/apps/manual_installations/speech2text/" .. version .. "/env/bin/"

prepend_path("PATH", speech2text)
prepend_path("PATH", conda_env)

local hf_home = "/scratch/shareddata/speech2text"
local torch_home = "/scratch/shareddata/speech2text"
local whisper_cache = "/scratch/shareddata/dldata/huggingface-hub-cache/hub/"
local pyannote_config = "/share/apps/manual_installations/speech2text/" .. version .. "/pyannote/config.yml"
local numba_cache = "/tmp" 
local mplconfigdir = "/tmp"

pushenv("HF_HOME", hf_home)
pushenv("TORCH_HOME", torch_home)
pushenv("XDG_CACHE_HOME", torch_home)
pushenv("WHISPER_CACHE", whisper_cache)
pushenv("PYANNOTE_CONFIG", pyannote_config)
pushenv("NUMBA_CACHE_DIR", numba_cache)
pushenv("MPLCONFIGDIR", mplconfigdir)

local speech2text_mem = "12G"
local speech2text_cpus_per_task = "6"
local speech2text_time = "24:00:00"
local speech2text_tmp = os.getenv("WRKDIR") .. "/.speech2text"

pushenv("SPEECH2TEXT_MEM", speech2text_mem)
pushenv("SPEECH2TEXT_CPUS_PER_TASK", speech2text_cpus_per_task)
pushenv("SPEECH2TEXT_TIME", speech2text_time)
pushenv("SPEECH2TEXT_TMP", speech2text_tmp)

pushenv("HF_HUB_OFFLINE", "1")

if mode() == "load" then
    LmodMessage("For more information, run 'module spider speech2text/" .. version .. "'")
end
 
