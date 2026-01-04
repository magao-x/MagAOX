import datetime
import logging
from typing import Union, Optional

import psycopg
from psycopg.sql import SQL, Identifier, Literal

from .records import *
from .config import *

log = logging.getLogger(__name__)

def connect():
    db = DbConfig()
    return db.connect()

DateTimeOrStr = Union[datetime.datetime,datetime.date,str]

def fetch(device: str, key: Optional[str] = None, ec: Optional[str]=None, start: Optional[DateTimeOrStr]=None, end: Optional[DateTimeOrStr]=None, conn: Optional[psycopg.Connection]=None, limit: Optional[int]=None):
    if conn is None:
        conn = connect()
    cur = conn.cursor()
    bounds = []
    date_criteria = ''
    if start is not None:
        date_criteria += ' and ts >= %s '
        bounds.append(start)
    if end is not None:
        date_criteria += ' and ts <= %s '
        bounds.append(end)
    date_criteria = SQL(date_criteria)
    if key is not None:
        message = SQL("    msg->>{key} AS {key_ident}").format(
            key=Literal(key),
            key_ident=Identifier(key),
        )
        message_criteria = SQL("AND msg->>{key} IS NOT NULL").format(key=Literal(key))
    else:
        message = SQL("msg")
        message_criteria = SQL("")
    ec_criteria = SQL("AND ec = {}").format(Literal(ec)) if ec is not None else SQL('')
    limit_criteria = SQL('LIMIT {}').format(Literal(limit)) if limit is not None else SQL('')
    query = SQL('''
SELECT
    ts,
    {message}
FROM
    telem
WHERE
    device = %s
{message_criteria}
{ec_criteria}
{date_criteria}
ORDER BY
    ts ASC
{limit_criteria}
''').format(
    message=message,
    message_criteria=message_criteria,
    date_criteria=date_criteria,
    ec_criteria=ec_criteria,
    limit_criteria=limit_criteria,
)
    variables = (device,) + tuple(bounds)
    log.debug(f"Executing {query} with variables {variables}")
    cur.execute(query, variables)
    return cur.fetchall()

def query_observations(start_dt=None, end_dt=None, title=None, email=None, conn: Optional[psycopg.Connection]=None, exact_title: bool=False):
    if conn is None:
        conn = connect()
    cur = conn.cursor()
    criteria = []

    if start_dt is not None or end_dt is not None:
        criteria.append(SQL("(tstzrange(start_ts, end_ts) && tstzrange({}, {}))").format(start_dt, end_dt))
    if title is not None:
        if not exact_title:
            criteria.append(SQL("obsname ILIKE {}").format(
                f"%{title}%"
            ))
        else:
            criteria.append(SQL("obsname = {}").format(title))
    if email is not None:
        criteria.append(SQL("email ILIKE {}").format(
            f"%{email}%"
        ))
    where_conditions = SQL(" AND ").join(criteria)
    query = SQL('''
SELECT * FROM observations
WHERE
{where_conditions}
ORDER BY start_ts ASC
''').format(
    where_conditions=where_conditions,
)
    log.debug(f"Executing {query}")
    cur.execute(query)
    return cur.fetchall()

def query_files(start_dt, end_dt, conn: Optional[psycopg.Connection]=None):
    if conn is None:
        conn = connect()
    cur = conn.cursor()
    # case where a file has been open before the obs start, closes before obs end:
    # - creation time will be < start_dt, modification time will be between start and end
    # case where a file was opened during the obs, closed before obs end:
    # - creation time will be between start and end
    # case where a file was opened before the obs start, closes after obs end:
    # - creation time will be < start_dt, mod time will be > end_dt
    # case where file was opened after obs start, closes after obs end:
    # - creation will be > start_dt, mod time will be > end_dt
    query = SQL('''
select * from file_origins where
    creation_time between {start_dt} and {end_dt}
    or
    modification_time between {start_dt} and {start_dt}
    or
    ((creation_time < {start_dt}) and (modification_time > {start_dt}))
''').format(start_dt=start_dt, end_dt=end_dt)
    log.debug(f"Executing {query}")
    cur.execute(query)
    return cur.fetchall()
