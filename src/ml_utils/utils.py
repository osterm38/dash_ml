"""
A module to store miscellaneous small utility functions that don't have a more appropriate place (yet).

"""
# IMPORTS
import logging
from pathlib import Path
from typing import Optional, Union
import zipfile


# FUNCTIONS
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

LOG = get_logger(__name__, 'DEBUG')

def zip_dir(in_path: Union[str, Path], out_path: Optional[Union[str, Path]] = None, overwrite: bool = True):
    """zip up in_path directory to out_path zip file (defaults to in_path.zip)"""
    LOG.debug(f'{in_path=}, {out_path=}')
    in_path = Path(in_path)
    assert in_path.is_dir(), f'expected a directory to zip, got {in_path=}'
    out_path = in_path.with_suffix('.zip') if out_path is None else Path(out_path)
    if out_path.is_file():
        if not overwrite:
            LOG.debug(f'WARNING: {out_path=} exists, skipping')
            return out_path
        else:
            LOG.debug(f'WARNING: {out_path=} exists, continuing by overwriting!')
    files = (p for p in in_path.glob('**/*') if p.is_file())
    with zipfile.ZipFile(out_path, mode='w', compression=zipfile.ZIP_DEFLATED) as fh:
        for f in files:
            LOG.debug(f'writing {f}, {f.relative_to(in_path.parent)}')
            fh.write(f, f.relative_to(in_path.parent))
    return out_path
    