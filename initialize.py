"""for initialization purposes"""

from pathlib import Path as path
cwd = path.cwd()
DATA_DIR = rf"{cwd}\data"

data_dir_path = path(DATA_DIR)

data_dir_path.mkdir(parents=True, exist_ok=True)