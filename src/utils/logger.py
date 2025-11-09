import logging
import sys
from pathlib import Path
try:
    from colorlog import ColoredFormatter
    _HAS_COLORLOG = True
except Exception:
    _HAS_COLORLOG = False

def get_logger(name: str, log_file: str = "logs/pipeline.log"):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    if _HAS_COLORLOG:
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s — %(name)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s",
                                      "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s",
                                      "%Y-%m-%d %H:%M:%S"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
