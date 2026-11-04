import re
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import typing
import warnings
import wave

import purepyindi2
import purepyindi2.client

try:
    _platform = os.uname().sysname
    if _platform == "Linux":
        _current_uid = os.getuid()
        os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{_current_uid}"
        os.environ["PULSE_SERVER"] = f"unix:/run/user/{_current_uid}/pulse/native"
    from piper import PiperVoice
    from piper.download_voices import download_voice
    import librosa
except ImportError:
    warnings.warn("Use 'pip install piper-tts librosa' to get speech synthesis")

log = logging.getLogger(__name__)

VOICES_DIR = Path("~/.local/share/piper-tts/voices").expanduser()
CACHE_DIR = Path("~/.cache/magao-x-tts").expanduser()

DEFAULT_VOICE_NAME = "en_GB-cori-high"

_LOADED_VOICES = {}


def contains_substitutions(text):
    return '{' in text and '}' in text

@dataclass
class Voice:
    model : 'PiperVoice'
    name : str

@dataclass(eq=True, frozen=True)
class Speech:
    voice: Voice
    text: str

    def __repr__(self):
        return f"{self.__class__.__name__}(text={self.text!r}, voice={self.voice.name!r})"

    @property
    def fingerprint(self) -> str:
        return hashlib.shake_128(f"{self.voice.name} {self.text}".encode("utf8")).hexdigest(10)

    def __eq__(self, other):
        return getattr(other, 'fingerprint', None) == self.fingerprint

@dataclass(eq=True, frozen=True)
class DynamicSpeech:
    text : str
    custom_voice_name : typing.Optional[str] = None

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.text)})"

    def __eq__(self, other):
        return (
            getattr(other, 'text', None) == self.text and
            getattr(other, 'custom_voice_name', None) == self.custom_voice_name
        )

    def to_speech(self, default_voice: Voice, indi_client: purepyindi2.client.IndiClient):
        if self.custom_voice_name:
            try:
                voice = load_voice(self.custom_voice_name)
            except ValueError as e:
                voice = default_voice
                log.warning(f"Got a request for a custom voice {self.custom_voice_name} but loading failed with {e}")
        else:
            voice = default_voice
        speech_text = self.text
        if contains_substitutions(speech_text):
            substitutables = re.findall(r"({[^}]+})", self.text)
            for sub in substitutables:
                indi_id = sub[1:-1]
                value = indi_client[indi_id]
                if hasattr(value, 'value'):
                    value = value.value
                log.debug(f"Replacing {repr(sub)} with {value=}")
                if value is not None:
                    try:
                        value = float(value)
                        value = "{:.1f}".format(value)
                    except (TypeError, ValueError):
                        value = str(value)
                    speech_text = speech_text.replace(sub, value)

        return Speech(voice, speech_text)

@dataclass(eq=True, frozen=True)
class Recording:
    path: str

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path!r})"
    def __eq__(self, other):
        return getattr(other, 'path', None) == self.path

PlaybackRequest = typing.Union[Speech, DynamicSpeech, Recording]

def load_voice(name=DEFAULT_VOICE_NAME, reload=False) -> Voice:
    if name in _LOADED_VOICES:
        if not reload:
            return _LOADED_VOICES[name]
        else:
            del _LOADED_VOICES[name]
    if not VOICES_DIR.exists():
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
    destfile = VOICES_DIR / f"{name}.onnx"
    if not destfile.exists():
        download_voice(name, VOICES_DIR)
    _LOADED_VOICES[name] = Voice(PiperVoice.load(destfile), name)
    return _LOADED_VOICES[name]


def get_or_create_speech_file(speech: Speech, cache):
    filename = f"{speech.fingerprint}.wav"
    if cache:
        if not CACHE_DIR.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fpath = CACHE_DIR / filename
        if (CACHE_DIR / filename).exists():
            return fpath
    else:
        fd, fpath = tempfile.mkstemp(".wav", "magao-x-tts")
        fpath = Path(fpath)
    with wave.open(fpath.as_posix(), "wb") as wav_file:
        speech.voice.model.synthesize_wav(speech.text, wav_file)
    return fpath

def play_text(text, voice=None):
    voice = voice or load_voice()
    play_speech(Speech(voice, text), cache=False)

def play_speech(speech, cache=True):
    speech_wav_path = get_or_create_speech_file(speech, cache=cache)
    play_audio_file(speech_wav_path)

def play_audio_file(audio_file: Path) -> None:
    if not audio_file.exists():
        raise FileNotFoundError(f"Can't find {audio_file}")
    match _platform:
        case 'Linux':
            subprocess.check_call(["paplay", audio_file.resolve().as_posix()])
        case 'Darwin':
            subprocess.check_call(["afplay", audio_file.resolve().as_posix()])

def duration_from_audio_file(audio_file: Path) -> float:
    return librosa.get_duration(path=audio_file)
