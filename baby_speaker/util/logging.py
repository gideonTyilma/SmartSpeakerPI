import logging, sys

def setup_logger(name="baby_speaker", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid dupes
        return logger
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(level)
    return logger
