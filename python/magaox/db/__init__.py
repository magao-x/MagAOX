from .records import *
from .config import *

def connect():
    db = DbConfig()
    return db.connect()
