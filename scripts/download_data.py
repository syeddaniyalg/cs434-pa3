import shutil
from pathlib import Path

import kagglehub
from tqdm import tqdm

from src import config
from src.data import unzip_in_place
from src.utils import make_dirs


def main():
    make_dirs()
    files = [config.META_FILE, config.ADJ_FILE] + [config.HIS_FILES[year] for year in config.YEARS]
    for name in tqdm(files, desc="LargeST files", unit="file"):
        target = config.RAW_DIR / name
        if target.exists():
            tqdm.write(f"already present: {target}")
            continue
        tqdm.write(f"downloading {name} from {config.KAGGLE_DATASET}")
        downloaded = Path(kagglehub.dataset_download(config.KAGGLE_DATASET, path=name))
        shutil.move(str(downloaded), str(target))
        unzip_in_place(target)
        tqdm.write(f"saved {target} ({target.stat().st_size / 1024 ** 2:.1f} MB)")


if __name__ == "__main__":
    main()
