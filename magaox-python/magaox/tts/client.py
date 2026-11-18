import dataclasses
import datetime
import pprint
import threading
import time
import queue
import logging
import purepyindi2
import pathlib

import xconf

from magaox.constants import DEFAULT_PREFIX
from . import core
from ..commands import Command

log = logging.getLogger(__name__)

RELOAD_WAIT_SEC = 5

@xconf.config
class IndiClientConfig:
    hostname : str = xconf.field(default='localhost', help='Hostname to connect to for INDI client')
    port : int = xconf.field(default=7624, help='Port to connect to for INDI client')

@xconf.config
class XAudioClient(Command):

    indi : IndiClientConfig = xconf.field(default_factory=IndiClientConfig, help="Connection information")
    audible_alerts_device_name : str = xconf.field(default='maggieo_x', help='Device publishing audible alerts')
    recordings_path : pathlib.Path = xconf.field(
        default=DEFAULT_PREFIX / "config" / "personalities" / "data",
        help="Subdirectory of the default config path with audio recordings"
    )

    def __post_init__(self):
        self._last_utterance_ts : datetime.datetime = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
        self._playback_queue : queue.Queue = queue.Queue()
        self._muted = purepyindi2.OFF
        if not self.recordings_path.exists():
            raise FileNotFoundError(f"{self.recordings_path} does not exist")

    @classmethod
    def get_default_config_prefix(cls) -> pathlib.Path:
        return DEFAULT_PREFIX / "config"

    def main(self):
        playback_thread = threading.Thread(target=self.do_audio_playback, name='XAudioClient-playback', daemon=True)
        playback_thread.start()

        # Connect to INDI but make the connection object by hand so
        # we can hook in for logging
        def log_connection(connection_status : purepyindi2.ConnectionStatus):
            log.info(f"Connection status changed to {connection_status}")
        conn = purepyindi2.transports.IndiTcpClientConnection(host=self.indi.hostname, port=self.indi.port, reconnect_automatically=True)
        conn.add_callback(purepyindi2.constants.TransportEvent.connection, log_connection)
        self._indi_client = purepyindi2.IndiClient(conn)
        self._indi_client.connect()
        self._indi_client.register_callback(
            self.handle_audible_alert,
            self.audible_alerts_device_name,
        )
        log.info("Registered audible alerts callback")
        while True:
            try:
                log.info("Starting XAudioClient...")
                self.run_client()
            except Exception as e:
                log.exception(e)
                time.sleep(RELOAD_WAIT_SEC)

    def do_audio_playback(self):
        log.info("Audio playback thread started")
        while req := self._playback_queue.get():
            self._last_utterance_ts = datetime.datetime.now(datetime.UTC)
            log.debug(f"Audio playback request: {req}")
            if self._muted is purepyindi2.ON:
                log.debug(f"Skipping because {self._muted=}")
                continue
            if isinstance(req, core.Speech):
                core.play_speech(req)
            elif isinstance(req, core.Recording):
                audio_file = self.recordings_path / req.path
                if audio_file.exists():
                    core.play_audio_file(audio_file)
                else:
                    log.error(f"{req.path} ({audio_file}) not found")

    def handle_audible_alert(self, msg: purepyindi2.messages.IndiDefSetDelMessage):
        assert msg.device == self.audible_alerts_device_name
        if isinstance(msg, purepyindi2.messages.DelProperty):
            log.debug(f"Deletion request, not doing anything {msg}")
            return
        if msg.name == 'mute':
            old_mute_state = self._muted
            self._muted = msg['toggle']
            log.debug(f"{msg.device}.mute.toggle updated. we had {old_mute_state}, now {self._muted}")
        if msg.name == 'playback':
            log.debug("\n" + str(msg))
            assert isinstance(msg, (purepyindi2.messages.SetTextVector, purepyindi2.messages.DefTextVector))
            log.debug(f"{msg.timestamp=} {self._last_utterance_ts=}")
            if msg.timestamp is not None and msg.timestamp > self._last_utterance_ts:
                log.debug(f"{msg.device}.playback_request timestamp {msg.timestamp} is newer than {self._last_utterance_ts=}")
                if msg['speech']:
                    new_audible_alert = core.Speech(core.load_voice(msg['voice']), msg['speech'])
                elif msg['file']:
                    new_audible_alert = core.Recording(self.recordings_path / msg['file'])
                else:
                    log.warning(f"Message lacked valid content but looked like an audio request: {msg}")
                    return
                self._playback_queue.put(new_audible_alert)
                log.debug(f"Enqueued new {new_audible_alert}")
            else:
                log.debug(f"Skipping request {msg['speech']=} {msg['file']=} because of {msg.timestamp=}")

    def handle_rule(self, msg: purepyindi2.messages.IndiDefSetDelMessage):
        log.debug("\n" + pprint.pformat(dataclasses.asdict(msg)))

    def run_client(self):
        if self._indi_client.connection.status is not purepyindi2.constants.ConnectionStatus.CONNECTED:
            log.info(f"Connection status is {self._indi_client.connection.status}, standing by")
            time.sleep(1)
        self._indi_client.get_properties_and_wait(self.audible_alerts_device_name)
        log.info(f"Got properties from {self.audible_alerts_device_name}")
        log.info("Running until exit")
        while True:
            time.sleep(1)
