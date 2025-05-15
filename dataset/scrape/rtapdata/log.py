import logging
import sys
from pathlib import Path
import os

# Set root logger to debug so that we have a way
# for any descendents of this logger to indicate
# that all fellow children should also be set to
# debug.
logging.getLogger('rtap-data').setLevel(logging.DEBUG)
logging.getLogger('rtap-data').propagate = False


class RTAPDataLogger(object):
    """
    A simple wrapper around the python logging
    tool to be used within RTAP data collection.
    """
    def __init__(self, logger_name, level=logging.INFO):
        logger_name = Path(logger_name).name
        self.logger = logging.getLogger(f'rtap.{logger_name}')
        self.log_path = Path(os.path.join(Path.home(), 'log'))
        self.log_path.mkdir(parents=True, exist_ok=True)

        if not self.logger.hasHandlers():
            self.formatter = logging.Formatter('%(asctime)s | %(levelname)7s | %(name)16s | %(filename)14s:%(lineno)4s | %(message)s')
            file_handler = logging.FileHandler(os.path.join(self.log_path, 'rtap-data.log'), 'a')
            file_handler.setFormatter(self.formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(level)
            LOG_LEVEL = os.getenv('LOG_LEVEL')

            if LOG_LEVEL is not None:
                self.logger.setLevel(LOG_LEVEL)

    def debug(self, message):
        self.logger.debug(message, stacklevel=2)

    def info(self, message):
        self.logger.info(message, stacklevel=2)

    def warning(self, message):
        self.logger.warning(message, stacklevel=2)

    def error(self, message):
        self.logger.error(message, stacklevel=2)

    def critical(self, message):
        self.logger.critical(message, stacklevel=2)
