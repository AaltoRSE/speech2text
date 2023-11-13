help_text = [[

This app does speech2text with diarization.

Example run: 

    speech2text audiofile.mp3 english

The speech2text app processes audiofile.mp3 and writes result files to a results/ subfolder (results/). 
Output filenames are the input filename with .txt and .csv extensions. For example, result files 
corresponding to audiofile.mp3 are results/audiofile.txt and results/audiofile.csv. The default output 
folder can be changed using the --output-dir option, for example:

    speech2text audiofile.mp3 english --output-dir my-results/

The audio files can contain speech in any supported language. Supported languages are:

afrikaans, arabic, armenian, azerbaijani, belarusian, bosnian, bulgarian, catalan, 
chinese, croatian, czech, danish, dutch, english, estonian, finnish, french, galician, 
german, greek, hebrew, hindi, hungarian, icelandic, indonesian, italian, japanese, 
kannada, kazakh, korean, latvian, lithuanian, macedonian, malay, marathi, maori, nepali,
norwegian, persian, polish, portuguese, romanian, russian, serbian, slovak, slovenian, 
spanish, swahili, swedish, tagalog, tamil, thai, turkish, ukrainian, urdu, vietnamese, 
welsh

If the input audio file contains speech in other than one of the supported languages, 
results will still be produced but may be nonsensical. 

If the input audio file is not .wav format, it will be converted within the script. 
The resulting .wav file is by default written to /tmp/. The original audio files 
are always retained.

Troubleshooting:

Requested Slurm resources can be adjusted using the following environment variables (default in parentheses):
- SPEECH2TEXT_MEM ("12G")
- SPEECH2TEXT_CPUS_PER_TASK (6)
- SPEECH2TEXT_TIME ("24:00:00")

]]

local version = "20231010"
whatis("Name : Aalto speech2text")
whatis("Version :" .. version)
help(help_text)

local speech2text = "/share/apps/manual_installations/speech2text/" .. version .. "/bin/"
local conda_env = "/share/apps/manual_installations/speech2text/" .. version .. "/env/bin/"

prepend_path("PATH", speech2text)
prepend_path("PATH", conda_env)

local hf_home = "/scratch/shareddata/speech2text"
local torch_home = "/scratch/shareddata/speech2text"
local whisper_cache = "/scratch/shareddata/openai-whisper/envs/venv-2023-03-29/models/"
local numba_cache = "/tmp" 
local mplconfigdir = "/tmp"

pushenv("HF_HOME", hf_home)
pushenv("TORCH_HOME", torch_home)
pushenv("XDG_CACHE_HOME", torch_home)
pushenv("WHISPER_CACHE", whisper_cache)
pushenv("NUMBA_CACHE_DIR", numba_cache)
pushenv("MPLCONFIGDIR", mplconfigdir)

local speech2text_mem = "12G"
local speech2text_cpus_per_task = "6"
local speech2text_time = "24:00:00"

pushenv("SPEECH2TEXT_MEM", speech2text_mem)
pushenv("SPEECH2TEXT_CPUS_PER_TASK", speech2text_cpus_per_task)
pushenv("SPEECH2TEXT_TIME", speech2text_time)

pushenv("HF_HUB_OFFLINE", "1")

if mode() == "load" then
    LmodMessage("For more information, run 'module spider speech2text/" .. version .. "'")
end
 
