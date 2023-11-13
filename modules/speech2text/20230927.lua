help_text = [[

This app does speech2text with diarization.

Example run: 

    speech2text data/

The speech2text app loops over all audio files in the target folder (data/) and 
writes result files to a results/ subfolder (data/results/). Output filenames for each 
input file are the input filename with .txt and .csv extensions. For example, result files 
corresponding to data/audiofile.wav are data/results/audiofile.txt and 
data/results/audiofile.csv. The default output folder can be changed using the
--output-dir option, for example:

    speech2text data/ --output-dir my-results/

The audio files can contain speech in any supported language. Supported languages are:

afrikaans, arabic, armenian, azerbaijani, belarusian, bosnian, bulgarian, catalan, 
chinese, croatian, czech, danish, dutch, english, estonian, finnish, french, galician, 
german, greek, hebrew, hindi, hungarian, icelandic, indonesian, italian, japanese, 
kannada, kazakh, korean, latvian, lithuanian, macedonian, malay, marathi, maori, nepali,
norwegian, persian, polish, portuguese, romanian, russian, serbian, slovak, slovenian, 
spanish, swahili, swedish, tagalog, tamil, thai, turkish, ukrainian, urdu, vietnamese, 
welsh

If input audio files contain speech in other than one of the supported languages, 
results will still be produced but are most probably nonsense. 

If one or more input audio files are not of .wav format, they will be converted to .wav files 
within the script. The resulting .wav files are by default removed after the script has 
been executed, but can be retained using option --keep-converted-files. The path to 
each converted file is the same as the original file with .wav extension: for example, 
data/audiofile.mp3 will be converted to data/audiofile.wav. The original audio files 
are always retained.

Troubleshooting:

Requested Slurm resources can be adjusted using the following environment variables (default in parentheses):
- SPEECH2TEXT_MEM ("12G")
- SPEECH2TEXT_CPUS_PER_TASK (8)
- SPEECH2TEXT_TIME ("12:00:00")

]]

local version = "20230927"
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
local speech2text_cpus_per_task = "8"
local speech2text_time = "72:00:00"

pushenv("SPEECH2TEXT_MEM", speech2text_mem)
pushenv("SPEECH2TEXT_CPUS_PER_TASK", speech2text_cpus_per_task)
pushenv("SPEECH2TEXT_TIME", speech2text_time)

pushenv("HF_HUB_OFFLINE", "1")

if mode() == "load" then
    LmodMessage("For more information, run 'module spider speech2text/" .. version .. "'")
end
 
