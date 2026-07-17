import glob
from purepyindi2 import device, properties, constants, messages
from purepyindi2.messages import DefNumber, DefSwitch, DefText
import sys
import logging
import xconf
import psycopg
from ...constants import DEFAULT_PREFIX
from ..config import BaseDbDeviceConfig
from .. import Telem, FileOrigin, UserLog
from .. import ingest

from ...utils import parse_iso_datetime_as_utc, creation_time_from_filename

import json
import orjson
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

log = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

class QueryCancellationWatchdog(threading.Thread):
    dt_sec : float = 0.1
    def __init__(self, conn: psycopg.Connection, timeout_sec=30.0):
        self.exit = False
        self.conn = conn
        self.timeout_sec = timeout_sec
        super().__init__(target=self.run)
    def complete(self):
        self.exit = True
    def run(self):
        total_wait_sec = 0.0
        while not self.exit and total_wait_sec < self.timeout_sec:
            time.sleep(self.dt_sec)
            total_wait_sec += self.dt_sec
        if self.exit:
            return
        else:
            self.conn.cancel_safe()
            raise TimeoutError()

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

def _run_logdump_thread(logger_name, logdump_dir, logdump_args, name, message_queue, record_class):
    # filter what content from user_logs gets put into db
    log = logging.getLogger(logger_name)
    while True:
        try:
            args = logdump_args + ('--dir='+logdump_dir, '-J', '-f', name)
            log.debug(f"Running logdump command {repr(' '.join(args))} for {name} in follow mode")
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            for line in p.stdout:
                message = record_class.from_json(name, line)
                message_queue.put(message)
            p.wait()  # stdout is over when the process exits
            if p.returncode != 0:
                raise RuntimeError(f"{name} logdump exited with {p.returncode} ({repr(' '.join(args))})")
        except Exception as e:
            glob_pattern = logdump_dir + f"/{name}/*/{name}_*"
            if len(glob.glob(glob_pattern + ".ndjson.gz")):
                log.info(f"Looks like {name} is a Python app; support is TODO")
                return
            if len(glob.glob(glob_pattern)):
                log.exception(f"Exception in log/telem follower for {name}")
            else:
                log.info(f"No files found for {name}, waiting for them to appear")
            while not len(glob.glob(glob_pattern)):
                time.sleep(RETRY_WAIT_SEC)

@xconf.config
class dbIngestConfig(BaseDbDeviceConfig):
    proclist : str = xconf.field(default="/opt/MagAOX/config/proclist_%s.txt", help="Path to process list file, %s will be replaced with the value of $MAGAOX_ROLE (or an empty string if absent from the environment)")
    query_timeout_sec : float = xconf.field(default=30.0, help="Number of seconds after which to (attempt to) cancel an insert query under the assumption the connection's gone bad")
    logdump_exe : str = xconf.field(default="/opt/MagAOX/bin/logdump", help="logdump (a.k.a. teldump) executable to use")

class dbIngest:
    config : dbIngestConfig
    telem_threads : list[tuple[str, threading.Thread]]
    telem_queue : queue.Queue
    fs_observer : BaseObserverSubclassCallable
    fs_queue : queue.Queue
    last_update_ts_sec : float
    startup_ts_sec : float
    records_since_startup : float
    _connections : dict[str, psycopg.Connection]
    _connections_to_attempt : set[str]

    #add user_log support here
    user_log_threads : list[tuple[str, threading.Thread]]
    user_log_queue : queue.Queue

    def launch_followers(self, dev):
        telem_args = self.log.name + '.' + dev, '/opt/MagAOX/telem', (self.config.logdump_exe, '--ext=.bintel'), dev, self.telem_queue, Telem
        telem_thread = threading.Thread(target=_run_logdump_thread, args=telem_args, daemon=True)
        telem_thread.start()
        self.log.debug(f"Watching {dev} for incoming telem")
        self.telem_threads.append((dev, telem_thread))

        if dev == "observers":
            ULog_args = self.log.name + '.' + dev, '/opt/MagAOX/logs', (self.config.logdump_exe, '--ext=.binlog'), dev, self.user_log_queue, UserLog
            user_log_thread = threading.Thread(target=_run_logdump_thread, args= ULog_args, daemon=True)
            user_log_thread.start()
            self.log.debug(f"Watching {dev} for incoming user logs")
            self.user_log_threads.append((dev, user_log_thread))

    def refresh_properties(self):
        self.properties['last_update']['timestamp'] = self.last_update_ts_sec
        self.update_property(self.properties['last_update'])
        self.properties['records']['since_startup'] = self.records_since_startup
        self.properties['records']['per_sec'] = self.records_since_startup / (time.time() - self.startup_ts_sec)
        self.update_property(self.properties['records'])

    def setup(self):
        self.last_update_ts_sec = time.time()
        self._connections = {}
        self._connections_to_attempt = set(self.config.databases.keys())
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

        self.user_log_queue = queue.Queue()
        self.user_log_threads = []

        self.telem_queue = queue.Queue()
        self.telem_threads = []
        for dev in device_names:
            self.launch_followers(dev)

        self.startup_ts_sec = time.time()

        self.fs_queue = queue.Queue()
        event_handler = NewXFilesHandler(self.config.hostname, self.fs_queue, self.log.name + '.fs_observer')
        self.fs_observer = Observer()
        for dirname in self.config.data_dirs:
            dirpath = self.config.common_path_prefix / dirname
            if not dirpath.exists():
                self.log.debug(f"No {dirpath} to watch")
                continue
            self.fs_observer.schedule(event_handler, dirpath, recursive=True)
            self.log.info(f"Watching {dirpath} for changes")
        self.fs_observer.start()

    def ingest_line(self, line):
        # avoid infinite loop of modifying log file and getting notified of the modification
        if self.log_file_name.encode('utf8') not in line:
            self.log.debug(line)

    def _ensure_connected(self):
        for configkey in self.config.databases.keys():
            if configkey in self._connections_to_attempt or self._connections[configkey].closed:
                connections_to_reattempt = set()
                for configkey in self._connections_to_attempt:
                    try:
                        self._connections[configkey].close()
                    except Exception:
                        pass
                    try:
                        self._connections[configkey] = self.config.databases[configkey].connect()
                        self.log.info(f"Connected to {configkey} db")
                    except Exception:
                        self.log.exception(f"Failed to connect to {configkey} ({self.config.databases[configkey]})")
                        connections_to_reattempt.add(configkey)
                self._connections_to_attempt = connections_to_reattempt

    def loop(self):
        self._ensure_connected()
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

        user_logs = []
        try:
            while rec := self.user_log_queue.get(timeout=0.1):
                user_logs.append(rec)
                self.records_since_startup += 1
        except queue.Empty:
            pass

        for connkey in self._connections:
            conn = self._connections[connkey]
            self.log.debug(f"Batching ingest for {connkey}")
            query_timeout = QueryCancellationWatchdog(conn, timeout_sec=self.config.query_timeout_sec)
            try:
                query_timeout.start()
                ingest.batch_telem(conn, telems)
                query_timeout.complete()
            except Exception as e:
                self.log.exception(f"Caught exception {e} in batch telem ingest, reconnecting {connkey} on next loop()")
                self._connections_to_attempt.add(connkey)
                continue
            finally:
                query_timeout.join()

            query_timeout = QueryCancellationWatchdog(conn, timeout_sec=self.config.query_timeout_sec)
            try:
                query_timeout.start()
                ingest.batch_file_origins(conn, fs_events)
                query_timeout.complete()
            except Exception as e:
                self.log.exception(f"Caught exception {e} in batch file origins ingest, reconnecting {connkey} on next loop()")
                self._connections_to_attempt.add(connkey)
                continue
            finally:
                query_timeout.join()

            query_timeout = QueryCancellationWatchdog(conn, timeout_sec=self.config.query_timeout_sec)
            try:
                query_timeout.start()
                ingest.batch_user_log(conn, user_logs)
                query_timeout.complete()
            except Exception as e:
                self.log.exception(f"Caught exception {e} in batch user log ingest, reconnecting {connkey} on next loop()")
                self._connections_to_attempt.add(connkey)
                continue
            finally:
                query_timeout.join()

        this_ts_sec = time.time()
        self.last_update_ts_sec = this_ts_sec
        self.refresh_properties()

main = dbIngest.console_app
