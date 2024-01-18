import gzip
import glob
import datetime
import os.path
import sys
from functools import partial
import logging
import subprocess
import psutil
from purepyindi2 import Device, transports
import toml
import typing
import xconf

# n.b. replaced with logger scoped to device name during device init
log = logging.getLogger()

@xconf.config
class BaseConfig:
    sleep_interval_sec : float = xconf.field(default=1.0, help="Main loop logic will be run every `sleep_interval_sec` seconds")

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
        raw_config = xconf._get_config_data(default_config_path, config_paths, settings_strs)
        if config_dict is not None:
            for key, value in config_dict.items():
                if key in raw_config:
                    old_val = raw_config[key]
                    log.info(f"Using provided value {value} for {key} which was set to {old_val} in the loaded config files")
            raw_config.update(config_dict)
        try:
            instance = xconf.from_dict(cls, raw_config)
        except (xconf.UnexpectedDataError, xconf.MissingValueError) as e:
            raise xconf.ConfigMismatch(e, raw_config)
        return instance

class MagAOXLogFormatter(logging.Formatter):
    def formatTime(self, record, datefmt):
        return datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f000')

def init_logging(logger : logging.Logger, destination, console_log_level, file_log_level, all_verbose):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if all_verbose:
        file_log_level = console_log_level = logging.DEBUG
        logger = root
    log_format = '%(asctime)s %(levelname)s %(message)s (%(name)s:%(funcName)s:%(lineno)d)'
    # logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(destination)

    logger.addHandler(file_handler)
    file_handler.setLevel(file_log_level)

    console = logging.StreamHandler()
    console.setLevel(console_log_level)
    logger.addHandler(console)
    logger.setLevel(min(file_log_level, console_log_level))

    formatter = MagAOXLogFormatter(log_format)
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.info(f"Logging to {destination}")

class XDevice(Device):
    prefix_dir : str  = "/opt/MagAOX"
    logs_dir : str = "logs"
    config_dir : str = "config"
    log : logging.Logger
    config : BaseConfig

    @classmethod
    @property
    def default_config_path(cls):
        return cls.prefix_dir + "/" + cls.config_dir + "/" + cls.__name__ + ".conf"

    @property
    def sleep_interval_sec(self):
        '''Sleep between executions of loop()

        Note this overrides the superclass class attribute with a read-only property
        referencing the config via `BaseConfig.sleep_interval_sec`'''
        return self.config.sleep_interval_sec

    @classmethod
    def load_config(cls, filenames=None, overrides=None):
        config_class : type = typing.get_type_hints(cls)['config']
        return config_class.from_config(cls.default_config_path, filenames, settings_strs=overrides)

    def _init_logs(self, verbose, all_verbose):
        global log
        log = self.log = logging.getLogger(self.name)
        log_dir = self.prefix_dir + "/" + self.logs_dir + "/" + self.name
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        log_file_name = f"{self.name}_{timestamp}.log"
        log_file_path = log_dir + "/" + log_file_name
        init_logging(
            log,
            log_file_path,
            console_log_level=logging.DEBUG if verbose else logging.INFO,
            file_log_level=logging.DEBUG,
            all_verbose=all_verbose)

        # GZip compress any existing text logs (ending in .log not .log.gz)
        for fn in glob.glob(log_dir + f"/{self.name}_*.log"):
            if fn.endswith(log_file_name):
                # don't gzip our empty log right after opening
                continue
            path_to_text_log = os.path.realpath(fn)
            return_code = subprocess.call(["gzip", path_to_text_log])
            if return_code != 0:
                log.error(f"Unable to compress {path_to_text_log} with gzip")
            else:
                log.debug(f"Compressed existing log: {fn}")

    def __init__(self, name, config, *args, verbose=False, all_verbose=False, **kwargs):
        fifos_root = self.prefix_dir + "/drivers/fifos"
        super().__init__(name, *args, connection_class=partial(transports.IndiFifoConnection, name=name, fifos_root=fifos_root), **kwargs)
        self.config = config
        self._init_logs(verbose, all_verbose)

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
        subprocess.check_call(["sudo", "/opt/MagAOX/bin/write_magaox_pidfile", str(thisproc.pid), self.name])

    def main(self):
        self.lock_pid_file()
        super().main()

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
        config = cls.load_config(args.config_file, args.vars)
        if args.dump_config:
            print(xconf.config_to_toml(config))
            sys.exit(0)
        cls(name=args.name, config=config, verbose=args.verbose, all_verbose=args.all_verbose).main()