import math
import os
import shutil
import zipfile

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config


def decode(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def normalize_id(value):
    text = str(decode(value)).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def unzip_in_place(path, chunk_mb=64):
    if path.suffix == ".npz" or not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        member = archive.infolist()[0]
        part = path.with_name(path.name + ".part")
        with archive.open(member) as src, open(part, "wb") as dst, \
                tqdm(total=member.file_size, unit="B", unit_scale=True, desc=f"unzipping {path.name}") as bar:
            while True:
                chunk = src.read(chunk_mb * 1024 ** 2)
                if not chunk:
                    break
                dst.write(chunk)
                bar.update(len(chunk))
    os.replace(part, path)
    return True


def prepare_raw_files():
    names = [config.META_FILE, config.ADJ_FILE] + [config.HIS_FILES[year] for year in config.YEARS]
    for name in names:
        path = config.RAW_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"missing {path}, run scripts.download_data first")
        if unzip_in_place(path):
            tqdm.write(f"extracted zipped download {path.name} ({path.stat().st_size / 1024 ** 3:.2f} GB)")
    return shutil.disk_usage(config.RAW_DIR).free


def find_frame_group(h5):
    found = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and "axis0" in obj and "axis1" in obj and "block0_values" in obj:
            found.append(name)

    h5.visititems(visit)
    if not found:
        raise ValueError("no pandas fixed-format frame found in the h5 file")
    return h5[found[0]]


def read_labels(dataset):
    return [normalize_id(v) for v in dataset[()]]


def read_time_index(dataset):
    values = dataset[()]
    kind = str(decode(dataset.attrs.get("kind", "datetime64")))
    unit = kind.split("[")[1].rstrip("]") if "[" in kind else "ns"
    times = pd.DatetimeIndex(values.astype(f"datetime64[{unit}]"))
    tz = dataset.attrs.get("tz", None)
    if tz is not None and str(decode(tz)) not in ("", "None"):
        times = times.tz_localize("UTC").tz_convert(str(decode(tz))).tz_localize(None)
    return times


def count_blocks(group):
    count = 0
    while f"block{count}_values" in group:
        count += 1
    return count


def read_sensor_columns(path, sensor_ids, chunk_rows=config.READ_CHUNK_ROWS):
    position = {sid: i for i, sid in enumerate(sensor_ids)}
    with h5py.File(path, "r") as h5:
        group = find_frame_group(h5)
        times = read_time_index(group["axis1"])
        out = np.full((len(times), len(sensor_ids)), np.nan, dtype=np.float32)
        found = np.zeros(len(sensor_ids), dtype=bool)
        for b in range(count_blocks(group)):
            items = read_labels(group[f"block{b}_items"])
            src = [j for j, item in enumerate(items) if item in position]
            if not src:
                continue
            dst = [position[items[j]] for j in src]
            values = group[f"block{b}_values"]
            time_rows = bool(values.attrs.get("transposed", values.shape[0] == len(times)))
            for start in tqdm(range(0, len(times), chunk_rows), desc=f"reading {path.name} block{b}", unit="chunk"):
                stop = min(start + chunk_rows, len(times))
                chunk = values[start:stop] if time_rows else values[:, start:stop].T
                out[start:stop, dst] = chunk[:, src]
            found[dst] = True
    return times, out, found


def load_ca_meta():
    meta = pd.read_csv(config.RAW_DIR / config.META_FILE)
    meta.columns = [c.strip().lower() for c in meta.columns]
    meta["ca_index"] = np.arange(len(meta))
    meta["id"] = meta["id"].map(normalize_id)
    return meta


def select_sd(meta):
    return meta[meta["district"] == config.SD_DISTRICT].reset_index(drop=True)


def load_sd_adjacency(sd_meta):
    ca_adj = np.load(config.RAW_DIR / config.ADJ_FILE, mmap_mode="r")
    idx = sd_meta["ca_index"].to_numpy()
    return np.asarray(ca_adj[np.ix_(idx, idx)], dtype=np.float32)


def full_time_grid(freq):
    return pd.date_range(config.TRAIN_START, config.TEST_END, freq=freq, inclusive="left")


def to_grid(times, values, freq):
    frame = pd.DataFrame(values, index=times)
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    return frame.reindex(full_time_grid(freq))


def resample(frame, method=config.RESAMPLE_METHOD):
    ratio = int(pd.Timedelta(config.FREQ) / pd.Timedelta(config.RAW_FREQ))
    data = frame.resample(config.FREQ).mean()
    if method == "sum":
        data = data * ratio
    return data.reindex(full_time_grid(config.FREQ))


def fill_missing(frame):
    mask = frame.notna().to_numpy()
    train_rows = frame.index < pd.Timestamp(config.VAL_START)
    train_means = frame[train_rows].mean()
    global_mean = float(np.nanmean(frame[train_rows].to_numpy()))
    filled = frame.ffill().fillna(train_means).fillna(global_mean)
    return filled.to_numpy(dtype=np.float32), mask


def load_processed():
    with np.load(config.PROCESSED_FILE, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def split_ranges(timestamps):
    times = pd.DatetimeIndex(timestamps)
    bounds = {
        "train": (config.TRAIN_START, config.VAL_START),
        "val": (config.VAL_START, config.TEST_START),
        "test": (config.TEST_START, config.TEST_END),
    }
    ranges = {}
    for name, (lo, hi) in bounds.items():
        idx = np.where((times >= pd.Timestamp(lo)) & (times < pd.Timestamp(hi)))[0]
        ranges[name] = (int(idx[0]), int(idx[-1]) + 1)
    return ranges


def train_correlations(flow, mask, timestamps):
    rows = slice(*split_ranges(timestamps)["train"])
    x = flow[rows].astype(np.float64)
    m = mask[rows]
    times = pd.DatetimeIndex(timestamps)[rows]
    slot = (times.dayofweek * config.STEPS_PER_DAY + times.hour * 4 + times.minute // 15).to_numpy()
    profile = pd.DataFrame(np.where(m, x, np.nan)).groupby(slot).mean().to_numpy()
    residual = np.where(m, x - profile[slot], 0.0)
    alive = m.any(axis=0)
    residual[:, alive] -= residual[:, alive].mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.corrcoef(x.T)
        local = np.corrcoef(residual.T)
    return np.nan_to_num(raw), np.nan_to_num(local)


def window_starts(start, stop):
    return np.arange(start, stop - config.INPUT_LEN - config.OUTPUT_LEN + 1)


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)

    @classmethod
    def fit(cls, values, mask):
        observed = values[mask]
        return cls(observed.mean(), observed.std())

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse(self, values):
        return values * self.std + self.mean


class TrafficData:
    def __init__(self, device):
        data = load_processed()
        self.device = device
        self.timestamps = pd.DatetimeIndex(data["timestamps"])
        self.sensor_ids = data["sensor_ids"]
        self.adj = data["adj"]
        self.lat = data["lat"]
        self.lng = data["lng"]
        self.plot_sensors = data["plot_sensors"]
        self.flow = data["flow"]
        self.observed = data["mask"]
        self.num_nodes = self.flow.shape[1]
        self.ranges = split_ranges(self.timestamps)
        train = slice(*self.ranges["train"])
        self.scaler = StandardScaler.fit(self.flow[train], self.observed[train])
        self.alive = self.observed[train].any(axis=0)
        self.series = torch.tensor(self.scaler.transform(self.flow), dtype=torch.float32, device=device)
        self.mask = torch.tensor(self.observed, dtype=torch.bool, device=device)
        self.starts = {name: torch.tensor(window_starts(*r), device=device) for name, r in self.ranges.items()}
        self.offsets = torch.arange(config.INPUT_LEN + config.OUTPUT_LEN, device=device)
        self.target_offsets = self.offsets[config.INPUT_LEN:]
        self.hours = torch.tensor(self.timestamps.hour.to_numpy(), dtype=torch.long, device=device)

    def target_hours(self, starts):
        return self.hours[starts[:, None] + self.target_offsets]

    def num_batches(self, split, batch_size):
        return math.ceil(len(self.starts[split]) / batch_size)

    def batches(self, split, batch_size, shuffle=False, seed=0):
        starts = self.starts[split]
        if shuffle:
            generator = torch.Generator().manual_seed(seed)
            starts = starts[torch.randperm(len(starts), generator=generator).to(self.device)]
        for i in range(0, len(starts), batch_size):
            s = starts[i:i + batch_size]
            idx = s[:, None] + self.offsets
            window = self.series[idx]
            x = window[:, :config.INPUT_LEN]
            y = self.scaler.inverse(window[:, config.INPUT_LEN:])
            m = self.mask[idx][:, config.INPUT_LEN:]
            yield x, y, m, s

    def summary(self):
        return {
            "num_nodes": self.num_nodes,
            "num_timestamps": len(self.timestamps),
            "scaler_mean": self.scaler.mean,
            "scaler_std": self.scaler.std,
            "sensors_never_observed_in_train": int((~self.alive).sum()),
            "splits": {
                name: {
                    "start": str(self.timestamps[r[0]]),
                    "end": str(self.timestamps[r[1] - 1]),
                    "timestamps": r[1] - r[0],
                    "windows": len(self.starts[name]),
                }
                for name, r in self.ranges.items()
            },
        }
