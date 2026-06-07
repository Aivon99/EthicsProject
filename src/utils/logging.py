import sys
import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a module-level logger with a consistent format.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger already exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger