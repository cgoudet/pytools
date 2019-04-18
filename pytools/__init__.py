import os
import pathlib
import logmatic
import logging

ROOT_DIRECTORY = pathlib.Path(os.path.dirname(__file__)) / '..'

# Get a logger by default and set its sensitivity to the maximum
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def define_logger(logger):
    """
    """
    # create a logging format as a JSON (works well for rabbit)
    formatter = logmatic.JsonFormatter(
        fmt="%(levelname) $(name) $(message)", extra={})

    # Create an object which redirect logs to a text file
    # Keep 10 files of 5MB for history
    # With tuning the logs can be redirected anywhere
    log_dir = ROOT_DIRECTORY/'logs'
    if not log_dir.exists():
        log_dir.mkdir()
    logfilename = log_dir/'happytal_libpython.log'
    handler = logging.handlers.RotatingFileHandler(
        str(logfilename), maxBytes=5000000, backupCount=10)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    # Multiple handlers with different types and sensitivity can be added to a unique logger
    logger.addHandler(handler)


define_logger(logger)
