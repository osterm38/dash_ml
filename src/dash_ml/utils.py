import logging
from typing import Optional 


def get_logger(
    # get local logger TODO: move to other module
    name: Optional[str] = None,
    level: str = 'DEBUG',
    stream: Optional[str] = None, #?
    fmt: Optional[str] = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
) -> logging.Logger:
    assert hasattr(logging, level)
    level = getattr(logging, level)
    formatter = logging.Formatter(fmt)
    ch = logging.StreamHandler(stream)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(ch)
    # TODO: what about to file?
    return logger
    