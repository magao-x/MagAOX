import json
import concurrent.futures
import xconf
import pathlib
from datetime import timezone
import datetime
import logging
import os
import os.path
import psycopg
import orjson
import socket
import typing
import subprocess
from tqdm import tqdm


import xconf
from ._base import BaseDbCommand

from magaox.db.config import DEFAULT_DATA_DIRS
from magaox.db import ingest
from magaox.db.records import Telem, FileIngestTime
from magaox.utils import parse_iso_datetime_as_utc, utcnow, xfilename_to_utc_timestamp

log = logging.getLogger(__name__)


@xconf.config
class Backfill(BaseDbCommand):
    """Process ``.bintel`` files found in data folders that don't already have ingest records
    and populate the ``telem`` table
    """

    logdump_exe: str = xconf.field(
        default="/opt/MagAOX/bin/logdump",
        help="logdump (a.k.a. teldump) executable to use",
    )
    parallel_jobs : int = xconf.field(default=10, help="Number of parallel workers to process individual .bintel files")

    def backfill_from_path(self, path):
        fname = os.path.basename(path)
        name, rest = fname.rsplit('_', 1)
        assert rest.endswith('.bintel'), 'not a bintel'
        args = [
            self.logdump_exe,
            "--ext=.bintel",
            "-J",
            "-F",
            path,
        ]
        log.debug(f"Launching logdump to read telemetry from {path} as NDJSON")
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.debug("Converting logdump output to records")
        records = []
        for line in p.stdout:
            assert line[0] == ord("{"), f"malformed line {line[0]=}"
            payload = json.loads(line)
            message = Telem(
                name,
                parse_iso_datetime_as_utc(payload["ts"]),
                payload["ec"],
                payload["msg"],
            )
            records.append(message)
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(
                f"{name} logdump exited with {p.returncode} ({repr(' '.join(args))})"
            )
        # pass to batch ingest
        log.debug(f"Ingesting {len(records)} record{'s' if len(records) != 1 else ''} into the database")
        conn = self.database.connect()
        try:
            with conn.transaction():
                cur = conn.cursor()
                ingest.batch_telem(cur, records)
                ingest.record_file_ingest_time(cur, FileIngestTime(
                    ts=xfilename_to_utc_timestamp(fname),
                    device=name,
                    ingested_at=utcnow(),
                    origin_host=self.hostname,
                    origin_path=path,
                ))
        finally:
            conn.close()
        return path

    def main(self):
        paths = ingest.identify_non_ingested_telem(
            self.database.cursor(), self.hostname
        )
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_jobs)
        futures = []
        log.info(f"Starting backfill tasks for {len(paths)} path{'s' if len(paths) != 1 else ''}")
        for fp in tqdm(paths):
            if os.path.exists(fp):
                futures.append(pool.submit(self.backfill_from_path, fp))
            else:
                log.debug(f"Skipping {fp} because the file does not exist")
        log.info("Ingesting files")
        pbar = tqdm(total=len(paths))
        for ft in concurrent.futures.as_completed(futures):
            log.debug(f"Finished {ft.result()}")
            pbar.update()
        pbar.close()
