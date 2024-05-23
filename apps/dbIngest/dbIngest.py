import glob
from purepyindi2 import device, properties, constants, messages
from purepyindi2.messages import DefNumber, DefSwitch, DefText
import sys
import logging
import xconf
from magaox.indi.device import XDevice
from magaox.db.config import BaseDeviceConfig
from magaox.db import Telem, FileOrigin
from magaox.db import ingest
from magaox.utils import parse_iso_datetime_as_utc, creation_time_from_filename

import json
import xconf
import subprocess
import queue
import socket
import threading
import pathlib
import time
import os.path
import os
import sys
import datetime
from datetime import timezone
from watchdog.observers import Observer, BaseObserverSubclassCallable
from watchdog.events import FileSystemEventHandler

class NewXFilesHandler(FileSystemEventHandler):
    def __init__(self, host, events_queue, log_name):
        self.host = host
        self.events_queue = events_queue
        self.log = logging.getLogger(log_name)

    def construct_message(self, stat_result, event, is_new_file=False):
        return FileOrigin(
            origin_host=self.host,
            origin_path=event.src_path,
            creation_time=creation_time_from_filename(event.src_path, stat_result=stat_result),
            modification_time=datetime.datetime.fromtimestamp(stat_result.st_mtime),
            size_bytes=stat_result.st_size,
        )

    def on_created(self, event):
        if event.is_directory:
            return
        try:
            stat_result = os.stat(event.src_path)
        except FileNotFoundError:
            return
        self.events_queue.put(self.construct_message(stat_result, event, is_new_file=True))

    def on_modified(self, event):
        if event.is_directory:
            return
        try:
            stat_result = os.stat(event.src_path)
        except FileNotFoundError:
            return
        self.events_queue.put(self.construct_message(stat_result, event, is_new_file=False))

RETRY_WAIT_SEC = 2
CREATE_CONNECTION_TIMEOUT_SEC = 2
EXIT_TIMEOUT_SEC = 2

def _run_logdump_thread(logger_name, logdump_dir, logdump_args, name, message_queue):
    log = logging.getLogger(logger_name)
    glob_pat = logdump_dir + f'/{name}_*'
    has_no_logs = len(glob.glob(glob_pat)) == 0
    if has_no_logs:
        log.debug(f"No matching files found for {glob_pat}")
    while True:
        while has_no_logs := len(glob.glob(glob_pat)) == 0:
            time.sleep(RETRY_WAIT_SEC)
        try:
            args = logdump_args + ('--dir='+logdump_dir, '-J', '-f', name)
            log.debug(f"Running logdump command {repr(' '.join(args))} for {name} in follow mode")
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            for line in p.stdout:
                message = Telem.from_json(name, line)
                message_queue.put(message)
            if p.returncode != 0:
                raise RuntimeError(f"{name} logdump exited with {p.returncode} ({repr(' '.join(args))})")
        except Exception as e:
            log.exception(f"Exception in log/telem follower for {name}")

@xconf.config
class dbIngestConfig(BaseDeviceConfig):
    proclist : str = xconf.field(default="/opt/MagAOX/config/proclist_%s.txt", help="Path to process list file, %s will be replaced with the value of $MAGAOX_ROLE (or an empty string if absent from the environment)")
    logdump_exe : str = xconf.field(default="/opt/MagAOX/bin/logdump", help="logdump (a.k.a. teldump) executable to use")

class dbIngest(XDevice):
    config : dbIngestConfig
    telem_threads : list[tuple[str, threading.Thread]]
    fs_observer : BaseObserverSubclassCallable
    telem_queue : queue.Queue
    fs_queue : queue.Queue
    last_update_ts_sec : float
    startup_ts_sec : float
    records_since_startup : float

    def launch_follower(self, dev):
        args = self.log.name + '.' + dev, '/opt/MagAOX/telem', (self.config.logdump_exe, '--ext=.bintel'), dev, self.telem_queue
        telem_thread = threading.Thread(target=_run_logdump_thread, args=args, daemon=True)
        telem_thread.start()
        self.log.debug(f"Watching {dev} for incoming telem")
        self.telem_threads.append((dev, telem_thread))

    def refresh_properties(self):
        self.properties['last_update']['timestamp'] = self.last_update_ts_sec
        self.update_property(self.properties['last_update'])
        self.properties['records']['since_startup'] = self.records_since_startup
        self.properties['records']['per_sec'] = self.records_since_startup / (time.time() - self.startup_ts_sec)
        self.update_property(self.properties['records'])

    def setup(self):
        self.last_update_ts_sec = time.time()
        self.records_since_startup = 0
        self.records_per_sec = 0.0
        last_update = properties.NumberVector(name="last_update", perm=constants.PropertyPerm.READ_ONLY)
        last_update.add_element(DefNumber(
            name="timestamp",
            _value=self.last_update_ts_sec,
            min=0.0, max=1e200, format='%f',
            step=1e-6,
        ))
        self.add_property(last_update)
        
        records = properties.NumberVector(name="records", perm=constants.PropertyPerm.READ_ONLY)
        records.add_element(DefNumber(
            name="per_sec",
            _value=0.0,
            min=0.0, max=1e200, format='%f',
            step=1e-6,
        ))
        records.add_element(DefNumber(
            name="since_startup",
            _value=0,
            min=0, max=1_000_000_000, format='%i',
            step=1,
        ))
        self.add_property(records)

        self.conn = self.config.database.connect()

        role = os.environ.get('MAGAOX_ROLE', '')
        proclist = pathlib.Path(self.config.proclist.replace('%s', role))
        if not proclist.exists():
            raise RuntimeError(f"No process list at {proclist}")

        device_names = set()

        with proclist.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                if line.strip()[0] == '#':
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    raise RuntimeError(f"Got malformed proclist line: {repr(line)}")
                device_names.add(parts[0])

        self.telem_queue = queue.Queue()
        self.telem_threads = []
        for dev in device_names:
            self.launch_follower(dev)
        
        self.startup_ts_sec = time.time()
        
        # rescan for inventory
        self.rescan_files()

        self.fs_queue = queue.Queue()
        event_handler = NewXFilesHandler(self.config.hostname, self.fs_queue, self.log.name + '.fs_observer')
        self.fs_observer = Observer()
        for dirpath in self.config.data_dirs:
            self.fs_observer.schedule(event_handler, dirpath, recursive=True)
            self.log.info(f"Watching {dirpath} for changes")
        self.fs_observer.start()

    def rescan_files(self):
        with self.conn.cursor() as cur:
            ingest.update_file_inventory(cur, self.config.hostname, self.config.data_dirs)
        self.log.info(f"Completed startup rescan of file inventory for {self.config.hostname}")

    def ingest_line(self, line):
        # avoid infinite loop of modifying log file and getting notified of the modification
        if self.log_file_name.encode('utf8') not in line:
            self.log.debug(line)

    def loop(self):
        telems = []
        try:
            while rec := self.telem_queue.get(timeout=0.1):
                telems.append(rec)
                self.records_since_startup += 1
        except queue.Empty:
            pass

        fs_events = []
        try:
            while rec := self.fs_queue.get(timeout=0.1):
                fs_events.append(rec)
                self.records_since_startup += 1
        except queue.Empty:
            pass

        with self.conn.transaction():
            cur = self.conn.cursor()
            ingest.batch_telem(cur, telems)
            ingest.batch_file_origins(cur, fs_events)

        this_ts_sec = time.time()
        self.last_update_ts_sec = this_ts_sec
        self.refresh_properties()
