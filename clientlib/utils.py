import os
import shutil
from pathlib import Path

def make_dirs(path, clean=True):
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    try:
        os.makedirs(path, exist_ok=True)
    except FileNotFoundError:
        return None
    return Path(path)