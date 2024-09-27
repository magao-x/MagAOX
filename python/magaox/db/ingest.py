import datetime
import logging
import os

from psycopg.types.json import Jsonb
import orjson

import psycopg
from psycopg import sql
from tqdm import tqdm

from .records import Telem, FileOrigin, FileReplica, FileIngestTime, UserLog
from ..utils import creation_time_from_filename, parse_iso_datetime_as_utc

log = logging.getLogger(__name__)

def batch_user_log(cur: psycopg.Cursor, records: list[UserLog]):
    """This will be where the logs will insert into user_logs table"""
    #for log in records:
    cur.execute('BEGIN')
    cur.executemany(f'''
INSERT INTO user_log (ts, device, ec, msg)
VALUES (%s, %s, %s::JSONB, %s)
ON CONFLICT (ts, ec) DO NOTHING;
''', [(rec.ts, rec.device, Jsonb(rec.msg, dumps=orjson.dumps), rec.ec)for rec in records])
    cur.execute('COMMIT')

def batch_telem(cur: psycopg.Cursor, records: list[Telem]):
    cur.execute("BEGIN")
    cur.executemany(f'''
INSERT INTO telem (ts, device, msg, ec)
VALUES (%s, %s, %s::JSONB, %s)
ON CONFLICT (device, ts) DO NOTHING;
''', [(rec.ts, rec.device, Jsonb(rec.msg, rec.ec, dumps=orjson.dumps)) for rec in records])
    cur.execute("COMMIT")

def batch_file_origins(cur: psycopg.Cursor, records: list[FileOrigin]):
    cur.execute("BEGIN")
    cur.executemany(f'''
INSERT INTO file_origins (origin_host, origin_path, creation_time, modification_time, size_bytes)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (origin_host, origin_path)
DO UPDATE SET modification_time = EXCLUDED.modification_time, size_bytes = EXCLUDED.size_bytes
''', [(rec.origin_host, rec.origin_path, rec.creation_time, rec.modification_time, rec.size_bytes) for rec in records])
    cur.execute("COMMIT")

def identify_new_files(cur: psycopg.Cursor, this_host: str, paths: list[str]):
    '''Returns the paths from ``paths`` that are not already part of the ``file_origins`` table'''
    if len(paths) == 0:
        return []
    cur.execute("BEGIN")
    try:
        # Create a temporary table with these paths to join against the db inventory
        cur.execute("CREATE TEMPORARY TABLE on_disk_files ( path VARCHAR(1024) )")
        query = f'''
    INSERT INTO on_disk_files (path)
    VALUES (%s)
    '''
        cur.executemany(query, [(x,) for x in paths])
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
    finally:
        cur.execute("ROLLBACK")  # ensure temp table is deleted
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

def update_file_inventory(cur: psycopg.Cursor, host: str, data_dirs: list[str]):
    """Update the file inventory with any untracked local files (if any)"""
    cur.execute("BEGIN")
    for prefix in data_dirs:
        for dirpath, dirnames, filenames in os.walk(prefix):
            log.info(f"Checking for new files in {dirpath}")
            new_files = identify_new_files(cur, host, [os.path.join(dirpath, fn) for fn in filenames])
            if len(new_files) == 0:
                continue
            else:
                log.info(f"Found {len(new_files)} new files in {dirpath}")
            origin_records = []
            for fn in tqdm(new_files):
                try:
                    stat_result = os.stat(fn)
                except FileNotFoundError:
                    log.info(f"Skipped {fn} (broken link?)")
                    continue
                except OSError as e:
                    log.info(f"Skipping {fn} because of error ({e})")
                origin_records.append(FileOrigin(
                    origin_host=host,
                    origin_path=fn,
                    creation_time=creation_time_from_filename(fn, stat_result=stat_result),
                    modification_time=datetime.datetime.fromtimestamp(stat_result.st_mtime),
                    size_bytes=stat_result.st_size,
                ))
            batch_file_origins(cur, origin_records)
    cur.execute("COMMIT")

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
