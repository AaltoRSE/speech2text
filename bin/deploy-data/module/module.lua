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
]]

whatis("Name : Aalto speech2text")
whatis("Version :" .. "<VERSION>")
help(help_text)

prepend_path("PATH", "<SPEECH2TEXT>")
prepend_path("PATH", "<CONDA_ENV>")

pushenv("HF_HOME", "<HF_HOME>")
pushenv("PYANNOTE_CACHE", "<PYANNOTE_CACHE>")
pushenv("TORCH_HOME", "<TORCH_HOME>")
pushenv("XDG_CACHE_HOME", "<TORCH_HOME>")
pushenv("PYANNOTE_CONFIG", "<PYANNOTE_CONFIG>")
pushenv("NUMBA_CACHE_DIR", "<NUMBA_CACHE_DIR>")
pushenv("MPLCONFIGDIR", "<MPLCONFIGDIR>")

pushenv("SPEECH2TEXT_MEM", "<SPEECH2TEXT_MEM>")
pushenv("SPEECH2TEXT_CPUS_PER_TASK", "<SPEECH2TEXT_CPUS_PER_TASK>")

pushenv("HF_HUB_OFFLINE", "1")

if mode() == "load" then
    LmodMessage("For more information, run 'module spider speech2text/" .. "<VERSION>" .. "'")
end
 
