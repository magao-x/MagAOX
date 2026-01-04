from collections.abc import Iterable
import datetime
from datetime import timezone
import logging
import os
import pathlib
import re
import itertools

from psycopg.types.json import Jsonb
import orjson

import psycopg
from psycopg import sql
from tqdm import tqdm

from .records import Telem, FileOrigin, FileReplica, FileIngestTime, UserLog
from ..utils import creation_time_from_filename, parse_iso_datetime_as_utc

log = logging.getLogger(__name__)

INGEST_IDENTIFY_FILES_BATCH_SIZE = 500
TELEM_BATCH_SIZE = 5_000
FILE_ORIGINS_BATCH_SIZE = 5_000

def batch_user_log(conn: psycopg.Connection, records: list[UserLog]):
    cur = conn.cursor()
    with conn.transaction():
        cur.executemany('''
            INSERT INTO user_log (ts, device, ec, msg)
            VALUES (%s, %s, %s, %s::JSONB)
            ON CONFLICT (ts, device) DO NOTHING;
            ''', [(rec.ts, rec.device, rec.ec, orjson.dumps(rec.msg).decode('utf8')) for rec in records])
        log.debug(f"Inserted {len(records)} user_logs into database")


def batch_telem(conn: psycopg.Connection, records: list[Telem]):
    cur = conn.cursor()
    for record_batch in itertools.batched(records, TELEM_BATCH_SIZE):
        with conn.transaction():
            cur.executemany(f'''
                INSERT INTO telem (ts, device, msg, ec)
                VALUES (%s, %s, %s::JSONB, %s)
                ON CONFLICT (device, ts) DO NOTHING;
                ''', [(rec.ts, rec.device, Jsonb(rec.msg, dumps=orjson.dumps), rec.ec) for rec in record_batch])

def batch_file_origins(conn: psycopg.Connection, records: list[FileOrigin]):
    cur = conn.cursor()
    for record_batch in itertools.batched(records, FILE_ORIGINS_BATCH_SIZE):
        with conn.transaction():
            cur.executemany(f'''
                INSERT INTO file_origins (origin_host, origin_path, creation_time, modification_time, size_bytes)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (origin_host, origin_path)
                DO UPDATE SET modification_time = EXCLUDED.modification_time, size_bytes = EXCLUDED.size_bytes
                ''', [(rec.origin_host, rec.origin_path, rec.creation_time, rec.modification_time, rec.size_bytes) for rec in record_batch])

def identify_new_files(conn: psycopg.Connection, this_host: str, paths: Iterable[pathlib.Path]):
    '''Returns the paths from ``paths`` that are not already part of the ``file_origins`` table'''
    if len(paths) == 0:
        return []
    cur = conn.cursor()
    with conn.transaction(force_rollback=True):
        # Create a temporary table with these paths to join against the db inventory
        cur.execute("CREATE TEMPORARY TABLE on_disk_files ( path VARCHAR(1024) )")
        query = f'''
    INSERT INTO on_disk_files (path)
    VALUES (%s)
    '''
        cur.executemany(query, [(x.as_posix(),) for x in paths])
        # execute_values(cur, query, )
        log.debug(f"Loaded {len(paths)} paths into temporary table for new file identification")

        # Identify paths without corresponding inventory rows
        q2 = sql.SQL('''
    WITH
        already_known_files AS (
            SELECT origin_path, origin_host FROM file_origins WHERE origin_host = %s
        )
    SELECT odf.path as path, akf.origin_path as origin_path
    FROM on_disk_files odf
    LEFT JOIN already_known_files akf ON
        odf.path = akf.origin_path
    WHERE akf.origin_path IS NULL
    ''').format()
        cur.execute(q2, (this_host,))
        log.debug(f"Found {cur.rowcount} new path{'s' if cur.rowcount != 1 else ''}")
        new_files = []
        for row in cur:
            new_files.append(row['path'])
    return new_files

#add non-ingested-userlogs?

def identify_non_ingested_telem(cur: psycopg.Cursor, host: str) -> list[str]:
    '''Use ``file_origins`` table to find ``.bintel`` file paths on the host
    ``host`` which need to be ingested'''
    # select file origins matching given hostname without ingest records
    # with extensions like '%.bintel'
    fns = []
    cur.execute('''
SELECT fi.origin_path as origin_path
FROM file_origins fi
LEFT JOIN file_ingest_times fit ON
    fi.origin_host = fit.origin_host AND
    fi.origin_path = fit.origin_path
WHERE
    (
        (
            fit.origin_host IS NULL AND
            fit.origin_path IS NULL
        ) OR (
            fit.ingested_at < fi.modification_time
        )
    )
    AND
    fi.origin_host = %s AND
    fi.origin_path LIKE '%%.bintel'
;
''', (host,))
    for row in cur:
        fns.append(row['origin_path'])
    return fns

def update_file_inventory(conn: psycopg.Connection, host: str, data_dirs: list[pathlib.Path],
                          ignored_file_patterns: list[str], ignored_directory_patterns: list[str]):
    """Update the file_origins table for a database pointed to by `cur` with untracked local files (if any)"""
    file_pattern = re.compile('|'.join(ignored_file_patterns))
    dir_pattern = re.compile('|'.join(ignored_directory_patterns))
    for prefix in data_dirs:
        for fpaths in itertools.batched(filter(lambda x: x.is_file(), prefix.glob("**")), INGEST_IDENTIFY_FILES_BATCH_SIZE):
            new_files = identify_new_files(conn, host, fpaths)
            if len(new_files) == 0:
                log.debug(f"Found zero new files from {fpaths}")
                continue
            with conn.transaction():
                log.info(f"Inventorying {len(new_files)} new files")
                log.debug("\n".join(new_files))
                origin_records = []
                for fn in tqdm(new_files):
                    if file_pattern.match(fn):
                        log.debug(f"Skipping {fn} because it matches the ignored files pattern")
                        continue
                    log.info(f"Inventorying {len(new_files)} new files")
                    log.debug("\n".join(new_files))
                    origin_records = []
                    for fn in tqdm(new_files):
                        if file_pattern.match(fn):
                            log.debug(f"Skipping {fn} because it matches the ignored files pattern")
                            continue
                        try:
                            stat_result = os.stat(fn)
                            origin_records.append(FileOrigin(
                                origin_host=host,
                                origin_path=fn,
                                creation_time=creation_time_from_filename(fn, stat_result=stat_result),
                                modification_time=datetime.datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc),
                                size_bytes=stat_result.st_size,
                            ))
                        except FileNotFoundError:
                            log.info(f"Skipped {fn} (broken link?)")
                            continue
                        except OSError as e:
                            log.info(f"Skipping {fn} because of error ({e})")
                    batch_file_origins(conn, origin_records)

def record_file_ingest_time(cur: psycopg.Cursor, rec : FileIngestTime):
    cur.execute("BEGIN")
    cur.execute(
"""
INSERT INTO file_ingest_times (ts, device, ingested_at, origin_host, origin_path)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (ts, device)
DO UPDATE SET ingested_at = EXCLUDED.ingested_at
""",
        (rec.ts, rec.device, rec.ingested_at, rec.origin_host, rec.origin_path)
    )
    cur.execute("COMMIT")
