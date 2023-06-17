from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import Final

__all__ = ["get_stream_logger", "add_file_handler"]

DEFAULT_FORMAT: Final[
    str
] = "[%(levelname)s] %(asctime)s - %(pathname)s : %(lineno)d : %(funcName)s : %(message)s"


def get_stream_logger(level: int = INFO, format: str = DEFAULT_FORMAT) -> Logger:
    logger = getLogger()
    logger.setLevel(level)

    handler = StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(Formatter(format))
    logger.addHandler(handler)
    return logger


def add_file_handler(
    logger: Logger, filename: str, level: int = INFO, format: str = DEFAULT_FORMAT
) -> Logger:
    handler = FileHandler(filename=filename)
    handler.setLevel(level)
    handler.setFormatter(Formatter(format))
    logger.addHandler(handler)
