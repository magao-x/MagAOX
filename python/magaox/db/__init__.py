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

def fetch(device: str, key: str, start: Optional[DateTimeOrStr]=None, end: Optional[DateTimeOrStr]=None, conn: Optional[psycopg.Connection]=None, limit: Optional[int]=None):
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
    limit_criteria = SQL('LIMIT {}').format(Literal(limit)) if limit is not None else SQL('')
    query = SQL('''
SELECT
    ts,
    msg->>{key} AS {key_ident}
FROM
    telem
WHERE
    device = %s
    AND msg->>{key} IS NOT NULL
{date_criteria}
ORDER BY
    ts ASC
{limit_criteria}
''').format(
    key=Literal(key),
    key_ident=Identifier(key),
    date_criteria=date_criteria,
    limit_criteria=limit_criteria,
)
    variables = (device,) + tuple(bounds)
    log.debug(f"Executing {query} with variables {variables}")
    cur.execute(query, variables)
    return cur.fetchall()
