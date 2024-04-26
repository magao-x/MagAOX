import logging
import xconf
import coloredlogs
from .commands import (
    setup,
    inventory,
    backfill,
)

# Split out for use in worker startup if needed
def _configure_logging(level, first_party_loggers):
    # remove existing handlers
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)
    # apply verbosity
    for logger_name in first_party_loggers:
        pkglog = logging.getLogger(logger_name)
        pkglog.setLevel(level)
        # add colors (if a tty)
        coloredlogs.install(level=level, logger=pkglog)

class Dispatcher(xconf.Dispatcher):
    first_party_loggers = ['magaox.db', 'xconf']
    def configure_logging(self, level):
        _configure_logging(level, self.first_party_loggers)

DISPATCHER = Dispatcher([
    setup.Setup,
    inventory.Inventory,
    backfill.Backfill,
])

