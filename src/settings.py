# Supported languages

supported_languages = {
    "arabic": "ar",
    "armenian": "hy",
    "bulgarian": "bg",
    "catalan": "ca",
    "chinese": "zh",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "german": "de",
    "greek": "el",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "kazakh": "kk",
    "korean": "ko",
    "latvian": "lv",
    "lithuanian": "lt",
    "malay": "ms",
    "marathi": "mr",
    "nepali": "ne",
    "norwegian": "no",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "spanish": "es",
    "swedish": "sv",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "vietnamese": "vi",
}

supported_languages_reverse = {value: key for key, value in supported_languages.items()}

supported_languages_pretty = ", ".join(
    [f"{lang} ({short})" for lang, short in supported_languages.items()]
)


# Wav2Vec models
wav2vec_models = {
    "hy": "infinitejoy/wav2vec2-large-xls-r-300m-armenian",
    "bg": "infinitejoy/wav2vec2-large-xls-r-300m-bulgarian",
    "et": "anton-l/wav2vec2-large-xlsr-53-estonian",
    "gl": "infinitejoy/wav2vec2-large-xls-r-300m-galician",
    "is": "language-and-voice-lab/wav2vec2-large-xlsr-53-icelandic-ep30-967h",
    "id": "indonesian-nlp/wav2vec2-large-xlsr-indonesian",
    "kk": "aismlv/wav2vec2-large-xlsr-kazakh",
    "lv": "infinitejoy/wav2vec2-large-xls-r-300m-latvian",
    "lt": "DeividasM/wav2vec2-large-xlsr-53-lithuanian",
    "ms": "gvs/wav2vec2-large-xlsr-malayalam",
    "mr": "infinitejoy/wav2vec2-large-xls-r-300m-marathi-cv8",
    "ne": "Harveenchadha/vakyansh-wav2vec2-nepali-nem-130",
    "ro": "anton-l/wav2vec2-large-xlsr-53-romanian",
    "sr": "dnikolic/wav2vec2-xlsr-530-serbian-colab",
    "sk": "infinitejoy/wav2vec2-large-xls-r-300m-slovak",
    "sl": "infinitejoy/wav2vec2-large-xls-r-300m-slovenian",
    "sv": "KBLab/wav2vec2-large-xlsr-53-swedish",
    "th": "sakares/wav2vec2-large-xlsr-thai-demo",
    "fi": "Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2",
}

# Whisper models
available_whisper_models = ["large-v2", "large-v3"]
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_COMPUTE_DEVICE = "cuda"
