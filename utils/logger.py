import logging
import os


def get_logger(name: str = "run", log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%m%d %H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger