import logging
import xconf
import coloredlogs
from magaox.db.commands import XTELEMDB_COMMANDS
from magaox.quicklook.commands import XQUICKLOOK_COMMANDS
from magaox.commands import Dispatcher

XTELEMDB = Dispatcher(XTELEMDB_COMMANDS)
XQUICKLOOK = Dispatcher(XQUICKLOOK_COMMANDS)
