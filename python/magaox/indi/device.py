import datetime
import os.path
import sys
from functools import partial
import logging
import subprocess
import psutil
from purepyindi2 import Device, transports
import toml

log = logging.getLogger(__name__)

class XDevice(Device):
    prefix_dir : str  = "/opt/MagAOX"
    logs_dir : str = "logs"
    config_dir : str = "config"
    log : logging.Logger
    config : dict

    def _init_config(self):
        config_file = self.prefix_dir + "/" + self.config_dir + "/" + self.name + ".conf"
        try:
            with open(config_file, 'r') as fh:
                self.config = toml.loads(fh)
        except Exception:
            log.exception(f"Could not load the config file (tried {config_file})")
            self.config = {}

    def _init_logs(self, verbose, all_verbose):
        self.log = logging.getLogger(self.name)
        log_dir = self.prefix_dir + "/" + self.logs_dir + "/" + self.name + "/"
        os.makedirs(log_dir, exist_ok=True)
        self.log.debug(f"Made (or found) {log_dir=}")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        log_file_path = log_dir + "/" + f"{self.name}_{timestamp}.log"
        log_format = '%(filename)s:%(lineno)d: [%(levelname)s] %(message)s'
        logging.basicConfig(
            level='INFO',
            filename=log_file_path,
            format=log_format
        )
        if verbose:
            self.log.setLevel(logging.DEBUG)
        if all_verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        # Specifying a filename results in no console output, so add it back
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(console)
        formatter = logging.Formatter(log_format)
        console.setFormatter(formatter)
        self.log.info(f"Logging to {log_file_path}")

    def __init__(self, name, *args, verbose=False, all_verbose=False, **kwargs):
        fifos_root = self.prefix_dir + "/drivers/fifos"
        super().__init__(name, *args, connection_class=partial(transports.IndiFifoConnection, name=name, fifos_root=fifos_root), **kwargs)
        self._init_config()
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
        parser = argparse.ArgumentParser()
        parser.add_argument('-n', '--name', help="Device name for INDI")
        parser.add_argument('-v', '--verbose', action='store_true', help="Set device log level to DEBUG")
        parser.add_argument('-a', '--all-verbose', action='store_true', help="Set global log level to DEBUG")
        args = parser.parse_args()
        cls(name=args.name, verbose=args.verbose, all_verbose=args.all_verbose).main()