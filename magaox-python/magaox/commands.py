import sys
import argparse
import xconf
import coloredlogs
import logging
from xconf import Command as XconfCommand

LOGGER_NAMES = ['magaox', 'xconf']

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
    first_party_loggers: list[str] = LOGGER_NAMES
    def configure_logging(self, level):
        _configure_logging(level, self.first_party_loggers)

class Command(XconfCommand):
    @classmethod
    def run(cls):
        parser = argparse.ArgumentParser(add_help=False)
        xconf.add_subparser_arguments(parser)
        args = parser.parse_args()
        if args.help:
            xconf.print_help(cls, parser)
            sys.exit(0)
        _configure_logging('DEBUG' if args.verbose else 'WARNING', LOGGER_NAMES)
        command = cls.from_args(args)
        if args.dump_config:
            print(xconf.config_to_toml(command))
            sys.exit(0)
        command._wrap_main()
