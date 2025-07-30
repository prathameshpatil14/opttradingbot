# utils/utils.py

import os
import yaml
import random
import numpy as np
import pandas as pd
import torch
import time
from datetime import datetime
from contextlib import contextmanager
import requests
from utils.logger import get_logger

logger = get_logger("Utils")

# ==== CONFIG LOADING ====

def load_yaml(path):
    """Load YAML config from path (safe mode)."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(obj, path):
    """Save dict/object as YAML to path."""
    with open(path, "w") as f:
        yaml.dump(obj, f)

def deep_get(d, keys, default=None):
    """Recursively get a value from nested dict d by a list of keys."""
    for key in keys:
        if d is None or key not in d:
            return default
        d = d[key]
    return d

def merge_dicts(a, b):
    """Recursively merge dict b into dict a."""
    for k, v in b.items():
        if (k in a and isinstance(a[k], dict) and isinstance(v, dict)):
            merge_dicts(a[k], v)
        else:
            a[k] = v
    return a

# ==== REPRODUCIBILITY ====

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# ==== SAFE DIR/FILE OPS ====

def safe_mkdir(path):
    """Create directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def safe_remove(path):
    """Remove file if exists."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Failed to remove {path}: {e}")

def safe_read_csv(path, **kwargs):
    """Safe pandas read_csv with logging."""
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        logger.error(f"Failed to read CSV: {path}: {e}")
        return pd.DataFrame()

def safe_write_csv(df, path, **kwargs):
    """Safe pandas to_csv with logging."""
    try:
        safe_mkdir(os.path.dirname(path))
        df.to_csv(path, **kwargs)
        logger.info(f"Wrote CSV: {path}")
    except Exception as e:
        logger.error(f"Failed to write CSV: {path}: {e}")

# ==== CHECKPOINTING ====

def save_checkpoint(obj, path):
    """Save PyTorch object (model/optimizer/state dict) to file."""
    safe_mkdir(os.path.dirname(path))
    torch.save(obj, path)
    logger.info(f"Saved checkpoint to {path}")

def load_checkpoint(path, device=None):
    """Load PyTorch object from checkpoint file."""
    if not os.path.exists(path):
        logger.warning(f"Checkpoint file not found: {path}")
        return None
    data = torch.load(path, map_location=device or "cpu")
    logger.info(f"Loaded checkpoint from {path}")
    return data

# ==== AUDIT LOGGING ====

def audit_log_row(path, row_dict, header=None):
    """
    Append a row (dict) to a CSV log file.
    Adds header if file is new.
    """
    safe_mkdir(os.path.dirname(path))
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        used_header = header or list(row_dict.keys())
        if write_header:
            f.write(",".join(used_header) + "\n")
        line = ",".join(str(row_dict.get(h, "")) for h in used_header)
        f.write(line + "\n")

# ==== TIMER/PROFILING ====

@contextmanager
def timer(name="block"):
    """Context manager for timing code blocks."""
    t0 = datetime.now()
    yield
    t1 = datetime.now()
    logger.info(f"Timer [{name}]: {(t1-t0).total_seconds():.3f}s")

# ==== NUMPY/DF HELPERS ====

def to_numpy(x):
    """Convert tensor or list to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)

def last_n(arr, n):
    """Return last n elements from list/array."""
    arr = np.asarray(arr)
    assert n >= 0
    return arr[-n:] if len(arr) >= n else arr

def normalize_series(s, window=30):
    """Rolling z-score normalization of a pandas Series."""
    return (s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-8)

# ==== DATE/TIME HELPERS ====

def today_str():
    """Returns today's date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")

def now_str():
    """Returns current datetime as string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ==== GENERAL ====

def ensure_list(x):
    """Ensure x is a list."""
    if isinstance(x, list):
        return x
    elif x is None:
        return []
    else:
        return [x]

# ==== ERROR/EXCEPTION HANDLING ====

def safe_call(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), catching and logging all exceptions."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.error(f"safe_call exception in {fn.__name__}: {e}", exc_info=True)
        return None

# ==== TIME/OTP HELPERS ====

def current_utc_unixtime():
    """Return current UTC time as Unix timestamp using worldtimeapi.org.

    Falls back to local time if the request fails.
    """
    url = "http://worldtimeapi.org/api/timezone/Etc/UTC"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return int(resp.json().get("unixtime", time.time()))
    except Exception as e:
        logger.warning(f"Failed to fetch remote time: {e}")
    return int(time.time())


def totp_now(secret):
    """Generate a TOTP using remote UTC time."""
    ts = current_utc_unixtime()
    import pyotp  # local import to avoid mandatory dependency when unused
    return pyotp.TOTP(secret).at(ts)

