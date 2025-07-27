# utils/logger.py

import logging
import os
from logging.handlers import RotatingFileHandler

_LOGGER_CACHE = {}

def get_logger(
    name, 
    log_file="logs/bot.log", 
    level="INFO", 
    max_bytes=5_000_000, 
    backup_count=5
):
    
    global _LOGGER_CACHE
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # Avoid double logging from root

    if not logger.handlers:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s %(name)s: %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Rotating file handler
        fh = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _LOGGER_CACHE[name] = logger
    return logger
