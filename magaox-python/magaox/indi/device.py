import datetime
from datetime import timezone
import inspect
import gzip
import glob
import json
import os
import os.path
import re
import signal
import stat
import sys
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
class ResurrecteeConfig:
    timeout: list[int] = xconf.field(
        default_factory=lambda: [10],
        help=(
            "Heartbeat timeout vector in seconds. First element is runtime heartbeat timeout. "
            "Optional second element is a one-time startup heartbeat timeout."
        ),
    )


@xconf.config
class BaseConfig:
    common_path_prefix : pathlib.Path = xconf.field(default=DEFAULT_PREFIX, help="Prefix for all instrument data and config directories")
    sleep_interval_sec : float = xconf.field(default=1.0, help="Main loop logic will be run every `sleep_interval_sec` seconds")
    resurrectee: ResurrecteeConfig = xconf.field(default_factory=ResurrecteeConfig)

    _SECTION_RE = re.compile(r"^\s*\[(.+?)\]\s*$")
    _LEGACY_TIMEOUT_RE = re.compile(r"^(\s*timeout\s*=\s*)(\d+(?:\s*,\s*\d+)+)\s*(#.*)?$")

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

    @classmethod
    def _normalize_legacy_resurrectee_timeout_toml(cls, raw_text: str):
        in_resurrectee = False
        changed = False
        out_lines = []

        for line in raw_text.splitlines(keepends=True):
            section_match = cls._SECTION_RE.match(line)
            if section_match:
                in_resurrectee = section_match.group(1).strip().lower() == "resurrectee"

            if in_resurrectee:
                timeout_match = cls._LEGACY_TIMEOUT_RE.match(line)
                if timeout_match:
                    vals = [v.strip() for v in timeout_match.group(2).split(",") if v.strip()]
                    comment = timeout_match.group(3) or ""
                    newline = "\n" if line.endswith("\n") else ""
                    line = f"{timeout_match.group(1)}[{', '.join(vals)}]{comment}{newline}"
                    changed = True

            out_lines.append(line)

        return "".join(out_lines), changed

    @classmethod
    def _load_config_file_with_compat(cls, filepath: str):
        log.debug(f"Loading config from {filepath}")
        with open(filepath, "r") as fh:
            raw_text = fh.read()

        try:
            return toml.loads(raw_text)
        except toml.TomlDecodeError:
            normalized_text, changed = cls._normalize_legacy_resurrectee_timeout_toml(raw_text)
            if not changed:
                raise
            log.debug(
                "Using legacy [resurrectee] timeout compatibility parser for %s; "
                "prefer timeout = [a, b] for strict TOML.",
                filepath,
            )
            return toml.loads(normalized_text)

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
                loaded_config = cls._load_config_file_with_compat(config_file_path)
                raw_config.update(loaded_config)
        elif default_config_path is not None:
            try:
                raw_config = cls._load_config_file_with_compat(default_config_path)
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

class XDevice(Device):
    prefix_dir : str  = "/opt/MagAOX"
    logs_dir : str = "logs"
    config_dir : str = "config"
    telem_dir : str = "telem"
    log : logging.Logger
    config : BaseConfig
    log_file_name : str
    _hb_fd : typing.Optional[int]
    _hb_last_time : int
    _hb_broken_pipes : int
    _hb_broken_pipes_limit : int
    _resurrectee_timeout : tuple[int, ...]
    _shutdown_requested : bool
    _prev_sigusr2_handler : typing.Any

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

    def _normalize_resurrectee_timeout(self):
        timeout = tuple(int(v) for v in self.config.resurrectee.timeout)
        if len(timeout) == 0:
            timeout = (10,)
        if timeout[0] < 1:
            timeout = (10,) + timeout[1:]
        return timeout

    def _resurrectee_fifo_path(self):
        return self.prefix_dir + "/drivers/fifos/" + self.name + ".hb"

    def _open_heartbeat_fifo(self):
        fifoname = self._resurrectee_fifo_path()
        fdrd = None
        fdhb = None
        try:
            try:
                fdrd = os.open(fifoname, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            except FileNotFoundError:
                prev_mode = os.umask(0)
                try:
                    os.mkfifo(fifoname, 0o660)
                finally:
                    os.umask(prev_mode)
                fdrd = os.open(fifoname, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)

            fdstat = os.fstat(fdrd)
            if not stat.S_ISFIFO(fdstat.st_mode):
                raise RuntimeError(f"{fifoname} exists but is not a FIFO")

            fdhb = os.open(fifoname, os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            self._hb_fd = fdhb
            self._hb_last_time = 0
        except Exception:
            self._hb_fd = None
            self.log.exception("Failed to open heartbeat FIFO %s", fifoname)
            if fdhb is not None:
                try:
                    os.close(fdhb)
                except Exception:
                    pass
        finally:
            if fdrd is not None:
                try:
                    os.close(fdrd)
                except Exception:
                    pass

    def _close_heartbeat_fifo(self):
        if self._hb_fd is not None:
            try:
                os.close(self._hb_fd)
            except Exception:
                self.log.exception("Failed to close heartbeat FIFO fd=%s", self._hb_fd)
            finally:
                self._hb_fd = None

    def _send_heartbeat(self, offset_index=0):
        if self._hb_fd is None:
            return
        if offset_index >= len(self._resurrectee_timeout):
            return

        offset = self._resurrectee_timeout[offset_index]
        if offset < 0:
            return

        new_time = int(time.time()) + offset
        if new_time <= self._hb_last_time:
            return

        stimestamp = f"{new_time:09x}\n".encode("ascii")
        try:
            os.write(self._hb_fd, stimestamp)
            self._hb_last_time = new_time
            if self._hb_broken_pipes:
                self.log.warning("Recovered heartbeat FIFO writes for %s", self.name)
                self._hb_broken_pipes = 0
        except Exception as e:
            self._hb_last_time = 0
            self._hb_broken_pipes += 1
            if self._hb_broken_pipes <= self._hb_broken_pipes_limit:
                self.log.error(
                    "Heartbeat write failed fd=%s payload=%r err=%s",
                    self._hb_fd,
                    stimestamp.decode("ascii").rstrip("\n"),
                    repr(e),
                )

    def _sigusr2_handler(self, sig, frame):
        self.log.info("Caught signal %s; clean shutdown requested", signal.strsignal(sig))
        self._shutdown_requested = True

    def _install_signal_handlers(self):
        self._prev_sigusr2_handler = signal.getsignal(signal.SIGUSR2)
        signal.signal(signal.SIGUSR2, self._sigusr2_handler)

    def _restore_signal_handlers(self):
        try:
            signal.signal(signal.SIGUSR2, self._prev_sigusr2_handler)
        except Exception:
            self.log.exception("Failed to restore SIGUSR2 handler")

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
        import configparser
        role = os.environ.get("MAGAOX_ROLE", "")
        if not role:
            return None
        conf_path = os.path.join(cls.prefix_dir, cls.config_dir, f"is{role}.conf")
        try:
            cp = configparser.ConfigParser()
            cp.read(conf_path)
            return cp.get("indiserver", "f", fallback=None)
        except Exception:
            return None

    def __init__(self, name, config, *args, verbose=False, all_verbose=False, **kwargs):
        self._startup_time = datetime.datetime.now(timezone.utc)
        fifos_root = self.prefix_dir + "/drivers/fifos"
        conn_kwargs = dict(name=name, fifos_root=fifos_root)
        if 'indiserver_ctrl_path' in inspect.signature(transports.IndiFifoConnection.__init__).parameters:
            conn_kwargs['indiserver_ctrl_path'] = self._get_indiserver_ctrl_path()
        super().__init__(name, *args, connection_class=partial(transports.IndiFifoConnection, **conn_kwargs), **kwargs)
        self.config = config
        self._hb_fd = None
        self._hb_last_time = 0
        self._hb_broken_pipes = 0
        self._hb_broken_pipes_limit = 2
        self._resurrectee_timeout = self._normalize_resurrectee_timeout()
        self._shutdown_requested = False
        self._prev_sigusr2_handler = signal.SIG_DFL
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
            super().main()
        finally:
            self.unlock_pid_file()

    def run(self):
        self.client = client.IndiClient()
        self.client.connect()
        self._install_signal_handlers()
        self._open_heartbeat_fifo()
        try:
            startup_offset_idx = 1 if len(self._resurrectee_timeout) > 1 else 0
            self._send_heartbeat(startup_offset_idx)

            self.setup()
            self.send_all_properties()
            self._setup_complete = True

            while (
                self.connection.status is constants.ConnectionStatus.CONNECTED
                and not self._shutdown_requested
            ):
                # Tied to the app loop by design: if loop() blocks, heartbeats stop.
                self._send_heartbeat(0)
                self._wrap_loop()
                time.sleep(self.sleep_interval_sec)
        finally:
            self._close_heartbeat_fifo()
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
