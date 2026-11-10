import datetime
from datetime import timezone
import inspect
import gzip
import glob
import json
import os
import os.path
import signal
import socket
import stat
import sys
import threading
import time
from functools import partial
import logging
import subprocess
import psutil
import toml
from purepyindi2 import Device
from purepyindi2 import transports, messages, client, constants
from purepyindi2.properties import IndiProperty
import pathlib
import typing
import xconf
import atexit


from ..utils import XFILENAME_TIME_FORMAT_OUT
from ..constants import DEFAULT_PREFIX

# n.b. replaced with logger scoped to device name during device init
log = logging.getLogger()

@xconf.config
class SystemdConfig:
    '''systemd service settings for this app, read by xctrl reset, not by the app'''
    restart: typing.Optional[str] = xconf.field(default=None, help="Restart= policy, e.g. always or on-failure")
    restartSec: typing.Optional[float] = xconf.field(default=None, help="RestartSec=, seconds before restarting")
    startLimitBurst: typing.Optional[int] = xconf.field(default=None, help="StartLimitBurst=, starts allowed per interval")
    startLimitIntervalSec: typing.Optional[float] = xconf.field(default=None, help="StartLimitIntervalSec=, seconds")
    watchdogSec: typing.Optional[float] = xconf.field(default=None, help="WatchdogSec=, seconds without a watchdog notification before systemd restarts the app")
    timeoutStartSec: typing.Optional[float] = xconf.field(default=None, help="TimeoutStartSec=, seconds allowed from start to READY=1")
    timeoutStopSec: typing.Optional[float] = xconf.field(default=None, help="TimeoutStopSec=, seconds before SIGKILL on stop")
    killSignal: typing.Optional[str] = xconf.field(default=None, help="KillSignal=, e.g. SIGTERM or SIGINT")


def sd_notify(state : str) -> bool:
    '''Send a state notification such as "READY=1" or "WATCHDOG=1" to systemd, as sd_notify(3)
    does. Does nothing unless systemd started us with Type=notify. Returns True if sent.'''
    path = os.environ.get("NOTIFY_SOCKET")
    if not path or path[0] not in "/@":
        return False
    if path[0] == "@":
        path = "\0" + path[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(state.encode(), path)
        return True
    except OSError:
        return False


@xconf.config
class BaseConfig:
    common_path_prefix : pathlib.Path = xconf.field(default=DEFAULT_PREFIX, help="Prefix for all instrument data and config directories")
    sleep_interval_sec : float = xconf.field(default=1.0, help="Main loop logic will be run every `sleep_interval_sec` seconds")
    systemd: SystemdConfig = xconf.field(default_factory=SystemdConfig)

    @staticmethod
    def _deep_update(base: dict, override: dict):
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                BaseConfig._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    @staticmethod
    def _load_config_file(filepath: str):
        log.debug(f"Loading config from {filepath}")
        with open(filepath, "r") as fh:
            return toml.load(fh)

    @classmethod
    def from_config(
        cls,
        default_config_path : typing.Optional[str] = None,
        config_path_or_paths: typing.Union[str,list[str]] = None,
        config_dict: typing.Optional[dict] = None,
        settings_strs: typing.Optional[list[str]] = None,
    ):
        '''Initialize a class instance using config files from disk, and/or a dictionary
        of options, and/or overrides from the cli
        '''

        config_paths = []
        if isinstance(config_path_or_paths, str):
            config_paths.append(config_path_or_paths)
        elif isinstance(config_path_or_paths, list):
            config_paths.extend(config_path_or_paths)
        if settings_strs is None:
            settings_strs = []

        raw_config = {}
        if len(config_paths):
            for config_file_path in config_paths:
                loaded_config = cls._load_config_file(config_file_path)
                raw_config.update(loaded_config)
        elif default_config_path is not None:
            try:
                raw_config = cls._load_config_file(default_config_path)
            except FileNotFoundError:
                pass

        cli_overrides = xconf._get_config_data(None, [], settings_strs)
        cls._deep_update(raw_config, cli_overrides)

        if config_dict is not None:
            for key, value in config_dict.items():
                if key in raw_config:
                    old_val = raw_config[key]
                    log.info(f"Using provided value {value} for {key} which was set to {old_val} in the loaded config files")
            cls._deep_update(raw_config, config_dict)
        try:
            instance = xconf.from_dict(cls, raw_config)
        except (xconf.UnexpectedDataError, xconf.MissingValueError) as e:
            raise xconf.ConfigMismatch(e, raw_config)
        return instance


def log_level_to_label(levelno):
    if levelno >= logging.CRITICAL:
        return "CRIT "
    elif levelno >= logging.ERROR:
        return "ERR  "
    elif levelno >= logging.WARNING:
        return "WARN"
    elif levelno >= logging.INFO:
        return "INFO"
    else:
        return "DBG "

class MagAOXLogFormatter(logging.Formatter):
    def __init__(self, fmt='%(asctime)s %(levelname)s %(message)s (%(name)s:%(funcName)s:%(lineno)d)'):
        super().__init__(fmt=fmt)
    def formatTime(self, record, datefmt):
        return datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + '000'

class NDJSONLogFormatter(MagAOXLogFormatter):
    def __init__(self):
        super().__init__("%(message)s")
    def format(self, record: logging.LogRecord):
        timestamp = self.formatTime(record, self.datefmt)
        # {"ts": "2024-11-12T22:15:11.554323648", "prio": "INFO", "ec": "text_log", "msg": {"message": "PID (6585) locked."}}
        label = log_level_to_label(record.levelno)
        return json.dumps({
            "ts": timestamp,
            "prio": label or "",
            "ec": "text_log",
            "msg": {"message": self.formatMessage(record)},
        })

def init_logging(logger : logging.Logger, destination, temporary_log_path, console_log_level, file_log_level, all_verbose):
    root = logging.getLogger()
    root.setLevel(logging.WARN)
    if all_verbose:
        file_log_level = console_log_level = logging.DEBUG
        logger = root

    if os.path.exists(temporary_log_path):
        os.remove(temporary_log_path)

    temp_file_handler = logging.FileHandler(temporary_log_path)
    logger.addHandler(temp_file_handler)
    temp_file_handler.setLevel(console_log_level)

    gzipped_file_log_handler = GzipStreamHandler(destination)
    gzipped_file_log_handler.setLevel(console_log_level)
    logger.addHandler(gzipped_file_log_handler)

    console = logging.StreamHandler()
    console.setLevel(console_log_level)
    logger.addHandler(console)
    logger.setLevel(min(file_log_level, console_log_level))

    text_formatter = MagAOXLogFormatter()
    ndjson_formatter = NDJSONLogFormatter()
    console.setFormatter(text_formatter)
    temp_file_handler.setFormatter(text_formatter)
    gzipped_file_log_handler.setFormatter(ndjson_formatter)
    logger.info(f"Logging to {destination}, current human-readable log at: {temporary_log_path}")

LINE_BUFFERED = 1

class GzipStreamHandler(logging.StreamHandler):
    _gzipfile : gzip.GzipFile
    def __init__(self, destination):
        self._gzipfile = gzip.open(destination, mode='wt', encoding='utf-8')
        atexit.register(self._gzipfile.close)
        super().__init__(self._gzipfile)

    def __del__(self):
        self._gzipfile.close()

class IndiDeviceHandler(logging.Handler):
    def __init__(self, device: Device, *args, **kwargs):
        self.device = device
        super().__init__(*args, **kwargs)
    def emit(self, record: logging.LogRecord):
        if record.levelno >= self.level:
            level_text = log_level_to_label(record.levelno)
        else:
            return
        msg = level_text + record.getMessage()
        if self.device.connected:
            self.device.connection.send(messages.Message(message=msg, device=self.device.name, timestamp=datetime.datetime.now(timezone.utc)))

class XIndiFifoConnection(transports.IndiFifoConnection):
    '''FIFO transport for the MagAO-X indiserver, which XDevice announces itself to with
    "start NAME PID" and "alive NAME PID" lines on the indiserver control FIFO.'''

    def _register_with_indiserver(self):
        # XDevice announces instead, after locking its PID file and emptying the input FIFO
        pass

    def drain_input(self):
        '''Discard whatever is waiting in the input FIFO, sent to a previous instance.
        Returns the number of bytes discarded.'''
        fd = self.input_pipe.fileno()
        n = 0
        os.set_blocking(fd, False)
        try:
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                n += len(chunk)
        except BlockingIOError:
            pass
        finally:
            os.set_blocking(fd, True)
        return n


class XDevice(Device):
    prefix_dir : str  = "/opt/MagAOX"
    logs_dir : str = "logs"
    config_dir : str = "config"
    telem_dir : str = "telem"
    log : logging.Logger
    config : BaseConfig
    log_file_name : str
    _shutdown_requested : bool
    _prev_sigterm_handler : typing.Any
    _client_lock : threading.Lock
    _client_resubscribe : bool
    _client_pending : set
    _client_retry_delay : float
    _client_next_retry : float
    indiserver_announce_interval_sec : float = 5.0  # between "alive" announcements to indiserver, 0 disables
    _indi_ctrl_path : str
    _indi_announced : bool
    _indi_announce_time : float
    _indi_gate_end : float  # NewProperty messages are ignored until indiserver greets us or this time

    @classmethod
    def get_default_config_prefix(cls):
        # needed before config is loaded, so only cls attributes
        return cls.prefix_dir + "/" + cls.config_dir + "/"

    @classmethod
    def get_default_config_path(cls):
        # needed before config is loaded, so only cls attributes
        return cls.get_default_config_prefix() + cls.__name__ + ".conf"

    @property
    def sleep_interval_sec(self):
        '''Sleep between executions of loop()

        Note this overrides the superclass class attribute with a read-only property
        referencing the config via `BaseConfig.sleep_interval_sec`'''
        return self.config.sleep_interval_sec

    @classmethod
    def load_config(cls, filenames=None, overrides=None):
        config_class : type = typing.get_type_hints(cls)['config']
        return config_class.from_config(cls.get_default_config_path(), filenames, settings_strs=overrides)

    def _init_logs(self, verbose, all_verbose):
        global log
        log = self.log = logging.getLogger(self.name)
        log_dir = self.prefix_dir + "/" + self.logs_dir + "/" + self.name
        os.makedirs(log_dir, exist_ok=True)
        timestamp = self._startup_time.strftime(XFILENAME_TIME_FORMAT_OUT)
        self.log_file_name = f"{self.name}_{timestamp}.ndjson.gz"
        log_file_path = log_dir + "/" + self.log_file_name
        temporary_log_path = f"/tmp/{self.name}.log"
        init_logging(
            log,
            log_file_path,
            temporary_log_path,
            console_log_level=logging.DEBUG if verbose else logging.INFO,
            file_log_level=logging.DEBUG if verbose else logging.INFO,
            all_verbose=all_verbose)
        log.addHandler(IndiDeviceHandler(self, level=logging.INFO))

    def _init_telem(self):
        telem_dir = self.prefix_dir + "/" + self.telem_dir + "/" + self.name
        timestamp = self._startup_time.strftime(XFILENAME_TIME_FORMAT_OUT)
        telem_file_name = f"{self.name}_{timestamp}.ndjson.gz"
        os.makedirs(telem_dir, exist_ok=True)
        telem_file_path = telem_dir + "/" + telem_file_name
        self._telem_file = gzip.open(telem_file_path, mode='wt', encoding='utf8')
        atexit.register(self._telem_file.close)
        self.log.info(f"Telemetrying to {telem_file_path}")

    def _sigterm_handler(self, sig, frame):
        self.log.info("Caught signal %s; clean shutdown requested", signal.strsignal(sig))
        self._shutdown_requested = True

    def _install_signal_handlers(self):
        # systemd stops services with SIGTERM: shut down cleanly
        self._prev_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._sigterm_handler)

    def _restore_signal_handlers(self):
        try:
            signal.signal(signal.SIGTERM, self._prev_sigterm_handler)
        except Exception:
            self.log.exception("Failed to restore SIGTERM handler")

    def telem(self, event : str, message : typing.Union[str, dict[str, typing.Any]]):
        current_ts_str = datetime.datetime.now().isoformat() + "000"  # zeros for consistency with MagAO-X timestamps with ns
        payload = {
            "ts": current_ts_str,
            "prio": "TELM",
            "ec": event,
            "msg": message,
        }
        json.dump(payload, self._telem_file, default=str)
        self._telem_file.write('\n')

    def update_property(self, prop : IndiProperty):
        super().update_property(prop)
        self.telem('telem_indi_set', prop.to_serializable())

    def __del__(self):
        self._telem_file.close()

    @classmethod
    def _get_indiserver_ctrl_path(cls):
        '''The indiserver control FIFO: [indiserver] f in is{MAGAOX_ROLE}.conf if set, else
        indiserver.ctrl with the driver FIFOs, the indiserver default.'''
        import configparser
        role = os.environ.get("MAGAOX_ROLE", "")
        if role:
            conf_path = os.path.join(cls.prefix_dir, cls.config_dir, f"is{role}.conf")
            try:
                cp = configparser.ConfigParser()
                cp.read(conf_path)
                path = cp.get("indiserver", "f", fallback=None)
                if path:
                    return path
            except Exception:
                pass
        return os.path.join(cls.prefix_dir, "drivers", "fifos", "indiserver.ctrl")

    def __init__(self, name, config, *args, verbose=False, all_verbose=False, **kwargs):
        self._startup_time = datetime.datetime.now(timezone.utc)
        fifos_root = self.prefix_dir + "/drivers/fifos"
        conn_kwargs = dict(name=name, fifos_root=fifos_root)
        super().__init__(name, *args, connection_class=partial(XIndiFifoConnection, **conn_kwargs), **kwargs)
        self._indi_ctrl_path = self._get_indiserver_ctrl_path()
        self._indi_announced = False
        self._indi_announce_time = 0.0
        self._indi_gate_end = 0.0
        self.config = config
        self._shutdown_requested = False
        self._prev_sigterm_handler = signal.SIG_DFL
        self._client_lock = threading.Lock()  # guards _client_pending and its retry timing
        self._client_resubscribe = False  # re-send all watched-property requests from the main loop
        self._client_pending = set()  # watched (device, name) interests deleted and not yet defined again
        self._client_retry_delay = 0.0
        self._client_next_retry = 0.0
        self._init_logs(verbose, all_verbose)
        self._init_telem()

    def lock_pid_file(self):
        pid_dir = self.prefix_dir + f"/sys/{self.name}"
        thisproc = psutil.Process()
        pid_file = pid_dir + "/pid"
        pid = None
        if os.path.exists(pid_file):
            with open(pid_file) as fh:
                try:
                    pid = int(fh.read())
                    log.debug(f"Got {pid=} from {pid_file}")
                except Exception:
                    pass
        if pid is not None:
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                with proc.oneshot():
                    if proc.exe() == sys.executable and self.name in sys.argv:
                        log.error(f"Found process ID {pid}: {proc.cmdline()} [{proc.status()}]")
                        sys.exit(1)
        log.debug(f"Writing PID file with PID {thisproc.pid}")
        subprocess.check_call(["/opt/MagAOX/bin/write_magaox_pidfile", str(thisproc.pid), self.name])

    def handle_message(self, message : messages.IndiMessage):
        # A getProperties without a property name may come from a restarted indiserver, which no
        # longer knows what we watch, so re-send the client's requests from the main loop.
        # It also greets this instance after its start announcement.
        if (
            isinstance(message, messages.GetProperties)
            and message.name is None
            and message.device in (None, self.name)
        ):
            self._client_resubscribe = True
            self._indi_gate_end = 0.0
        elif (
            isinstance(message, typing.get_args(messages.IndiNewMessage))
            and time.monotonic() < self._indi_gate_end
        ):
            self.log.info("Ignoring %s.%s sent before indiserver greeted this instance", message.device, message.name)
            return
        super().handle_message(message)

    def _announce(self, verb):
        '''Write "verb name pid" to the indiserver control FIFO without blocking.
        "start" once when the input FIFO has been emptied, so indiserver registers us or resets
        its session with a previous instance, "alive" periodically so a restarted indiserver
        registers us again. Returns True if written.'''
        self._indi_announce_time = time.monotonic()
        line = f"{verb} {self.name} {os.getpid()}\n".encode()
        try:
            fd = os.open(self._indi_ctrl_path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                ok = os.write(fd, line) == len(line)
            finally:
                os.close(fd)
            err = None
        except OSError as e:
            ok, err = False, e
        # log only changes, a missing indiserver would otherwise be logged every time
        if ok != self._indi_announced or verb == "start":
            if ok:
                self.log.info("Announced %s on indiserver control FIFO %s", verb, self._indi_ctrl_path)
            else:
                self.log.info("Could not announce %s on indiserver control FIFO %s: %s", verb, self._indi_ctrl_path, err)
        self._indi_announced = ok
        return ok

    def _indiserver_upkeep(self):
        if self.indiserver_announce_interval_sec > 0 and time.monotonic() - self._indi_announce_time >= self.indiserver_announce_interval_sec:
            self._announce("alive")

    def _client_interests(self):
        # purepyindi2's IndiClient records the (device, name) pairs passed to get_properties() here
        return tuple(getattr(self.client, '_interested_properties', ()))

    def _handle_client_message(self, message : messages.IndiDefSetDelMessage):
        '''Mark watched properties of a deleted device as not received, and received again once defined.'''
        if isinstance(message, messages.DelProperty):
            matches = [
                (dev, name) for dev, name in self._client_interests()
                if dev == message.device and (message.name is None or name in (constants.ALL, message.name))
            ]
            if matches:
                with self._client_lock:
                    self._client_pending.update(matches)
                    self._client_retry_delay = 0.0
                    self._client_next_retry = 0.0
        elif isinstance(message, typing.get_args(messages.IndiDefMessage)):
            with self._client_lock:
                self._client_pending.discard((message.device, message.name))
                self._client_pending.discard((message.device, constants.ALL))

    def _send_client_get_properties(self, interests):
        for dev, name in interests:
            try:
                self.client.connection.send(messages.GetProperties(
                    device=None if dev is constants.ALL else dev,
                    name=None if name is constants.ALL else name,
                ))
            except Exception:
                self.log.exception("Failed to send getProperties for %s.%s", dev, name)

    def _resubscribe_client(self):
        '''Re-send all watched-property requests after a generic getProperties, and re-request
        deleted ones with backoff (1 s doubling to 60 s) until they are defined again.'''
        if not self.connected:
            return
        if self._client_resubscribe:
            self._client_resubscribe = False
            self._send_client_get_properties(self._client_interests())
            return
        now = time.monotonic()
        with self._client_lock:
            if not self._client_pending or now < self._client_next_retry:
                return
            pending = tuple(self._client_pending)
            self._client_retry_delay = min(max(1.0, 2 * self._client_retry_delay), 60.0)
            self._client_next_retry = now + self._client_retry_delay
        self._send_client_get_properties(pending)

    def unlock_pid_file(self):
        pid_file = self.prefix_dir + f"/sys/{self.name}/pid"
        try:
            os.remove(pid_file)
            self.log.info("PID file removed: %s", pid_file)
        except FileNotFoundError:
            # Already cleaned up (or never created) is fine.
            pass
        except Exception:
            self.log.exception("Failed to remove PID file %s", pid_file)

    def main(self):
        self.lock_pid_file()
        try:
            # only now that no other instance runs, empty our input FIFO and tell indiserver
            # this is a new instance, it greets us with a getProperties
            ndrained = self.connection.drain_input()
            if ndrained:
                self.log.info("Discarded %d bytes of INDI input sent to a previous instance", ndrained)
            if self._announce("start"):
                self._indi_gate_end = time.monotonic() + 10
            super().main()
        finally:
            self.unlock_pid_file()

    def run(self):
        self.client = client.IndiClient()
        self.client.connect()
        self.client.register_callback(self._handle_client_message)
        self._install_signal_handlers()
        try:
            self.setup()
            self.send_all_properties()
            self._setup_complete = True
            # startup is complete; under systemd the watchdog applies from here on
            sd_notify("READY=1")

            while (
                self.connection.status is constants.ConnectionStatus.CONNECTED
                and not self._shutdown_requested
            ):
                # Tied to the app loop by design: if loop() blocks, systemd's watchdog restarts us.
                sd_notify("WATCHDOG=1")
                self._indiserver_upkeep()
                self._resubscribe_client()
                self._wrap_loop()
                time.sleep(self.sleep_interval_sec)
        finally:
            sd_notify("STOPPING=1")
            self._restore_signal_handlers()

    @classmethod
    def console_app(cls):
        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        xconf.add_subparser_arguments(parser)
        parser.add_argument('-n', '--name', help="Device name for INDI")
        parser.add_argument('-a', '--all-verbose', action='store_true', help="Set global log level to DEBUG")
        args = parser.parse_args()
        config_class = typing.get_type_hints(cls)['config']
        if args.help:
            xconf.print_help(config_class, parser)
            sys.exit(0)

        config_files = [args.config_file if args.name is None else cls.get_default_config_path]
        if args.name is not None:
            config_files = [os.path.join(cls.get_default_config_prefix(), args.name + '.conf')]
        elif len(args.config_file):
            config_files = args.config_file
        else:
            config_files = None
        config = cls.load_config(config_files, args.vars)
        if args.dump_config:
            print(xconf.config_to_toml(config))
            sys.exit(0)
        cls(name=args.name, config=config, verbose=args.verbose, all_verbose=args.all_verbose).main()
