from dataclasses import dataclass
import os.path
import subprocess
import hashlib
import logging
import requests
from urllib.parse import urljoin

log = logging.getLogger(__name__)

def check_opentts_service_status(api_url):
    return requests.head(api_url).status_code == 200

def speak(ssml_text, default_voice, api_url, cache_dir):
    log.debug("Generating speech (or using cache)...")
    wav_path = ssml_to_wav(ssml_text, default_voice, api_url, cache_dir)
    log.debug("Playing audio...")
    play_wav(wav_path)
    log.debug("Speech spoken.")

def ssml_to_wav(ssml_text, default_voice, api_url, cache_dir):
    cache_key = hashlib.md5(ssml_text.encode('utf-8')).hexdigest()
    cache_filename = f"{default_voice}_{cache_key}.wav"
    destination_path = os.path.join(cache_dir, cache_filename)
    if not os.path.exists(destination_path):
        if not check_opentts_service_status(api_url):
            raise RuntimeError(f"No OpenTTS at {api_url}, cannot generate speech")
        endpoint = urljoin(api_url, "/api/tts")
        resp = requests.get(
            endpoint,
            params={
                'voice': default_voice,
                'text': ssml_text,
                'vocoder': 'high',
                'denoiserStrength': 0.03,
                'cache': False,
                'ssml': True,
                'ssmlNumbers': True,
                'ssmlDates': True,
                'ssmlCurrency': True,
            },
            timeout=10,
        )
        with open(destination_path, 'wb') as fh:
            fh.write(resp.content)
        log.debug(f"Cache written: {destination_path}")
    else:
        log.debug(f"Cache hit: {destination_path}")
    return destination_path

def play_wav(wav_path):
    subprocess.check_call(f"XDG_RUNTIME_DIR=/run/user/$(id -u xsup) pacat --rate=22050 --channels=1 {wav_path}", shell=True)