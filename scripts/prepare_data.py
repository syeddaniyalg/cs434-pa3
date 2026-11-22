import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.data import (fill_missing, full_time_grid, load_ca_meta, load_sd_adjacency, prepare_raw_files,
                      read_sensor_columns, resample, select_sd)
from src.utils import Logger, make_dirs, save_fig, save_json, set_plot_style


def load_raw(sensor_ids, log):
    frames = []
    found_any = np.zeros(len(sensor_ids), dtype=bool)
    for year in config.YEARS:
        path = config.RAW_DIR / config.HIS_FILES[year]
        times, values, found = read_sensor_columns(path, sensor_ids)
        log(f"{path.name}: {len(times)} rows, {times.min()} -> {times.max()}, sensors found {found.sum()}/{len(found)}")
        frames.append(pd.DataFrame(values, index=times))
        found_any |= found
    raw = pd.concat(frames)
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    raw = raw[(raw.index >= pd.Timestamp(config.TRAIN_START)) & (raw.index < pd.Timestamp(config.TEST_END))]
    return raw.reindex(full_time_grid(config.RAW_FREQ)), found_any


def longest_gaps(missing):
    gaps = np.zeros(missing.shape[1], dtype=int)
    for j in tqdm(range(missing.shape[1]), desc="longest missing gaps", leave=False):
        column = np.concatenate([[0], missing[:, j].astype(np.int8), [0]])
        edges = np.diff(column)
        starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
        gaps[j] = (stops - starts).max() if len(starts) else 0
    return gaps


def choose_plot_sensors(flow, mask):
    missing_rate = 1 - mask.mean(axis=0)
    candidates = np.where(missing_rate < 0.05)[0]
    if len(candidates) < len(config.PLOT_SENSOR_QUANTILES):
        candidates = np.argsort(missing_rate)[:max(50, len(config.PLOT_SENSOR_QUANTILES))]
    means = np.array([flow[mask[:, j], j].mean() if mask[:, j].any() else 0.0 for j in candidates])
    order = candidates[np.argsort(means)]
    chosen = []
    for q in config.PLOT_SENSOR_QUANTILES:
        idx = int(order[min(int(q * len(order)), len(order) - 1)])
        while idx in chosen:
            idx = int(order[(list(order).index(idx) + 1) % len(order)])
        chosen.append(idx)
    return np.array(chosen)


def missing_summary(raw_missing, missing15, flow_frame):
    per_sensor_5 = raw_missing.mean(axis=0)
    per_sensor_15 = missing15.mean(axis=0)
    gaps = longest_gaps(missing15)
    return {
        "missing_rate_5min": float(raw_missing.mean()),
        "missing_rate_15min": float(missing15.mean()),
        "missing_rate_15min_by_split": {
            "train": float(missing15[flow_frame.index < pd.Timestamp(config.VAL_START)].mean()),
            "val": float(missing15[(flow_frame.index >= pd.Timestamp(config.VAL_START))
                                   & (flow_frame.index < pd.Timestamp(config.TEST_START))].mean()),
            "test": float(missing15[flow_frame.index >= pd.Timestamp(config.TEST_START)].mean()),
        },
        "sensors_fully_missing_15min": int((per_sensor_15 == 1).sum()),
        "sensors_without_missing_15min": int((per_sensor_15 == 0).sum()),
        "sensor_missing_rate_15min_median": float(np.median(per_sensor_15)),
        "sensor_missing_rate_15min_max": float(per_sensor_15.max()),
        "sensors_over_10pct_missing_15min": int((per_sensor_15 > 0.1).sum()),
        "timestamps_all_sensors_missing_5min": int(raw_missing.all(axis=1).sum()),
        "timestamps_all_sensors_missing_15min": int(missing15.all(axis=1).sum()),
        "longest_gap_15min_steps_median": float(np.median(gaps)),
        "longest_gap_15min_steps_max": int(gaps.max()),
        "per_sensor_5": per_sensor_5,
        "per_sensor_15": per_sensor_15,
    }


def plot_week(raw, flow15, mask, sensors, sensor_ids, start, name):
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(days=7)
    fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 2.6 * len(sensors)), sharex=True)
    for ax, j in zip(np.atleast_1d(axes), sensors):
        raw_week = raw.loc[start:end - pd.Timedelta(config.RAW_FREQ), j]
        week15 = flow15.loc[start:end - pd.Timedelta(config.FREQ), j]
        observed = pd.Series(mask[:, j], index=flow15.index).loc[week15.index]
        ax.plot(raw_week.index, raw_week.values, color="tab:gray", lw=0.6, alpha=0.7, label="5-min raw")
        ax.plot(week15.index, week15.where(observed).values, color="tab:blue", lw=1.3, label="15-min resampled")
        filled = week15.where(~observed)
        if filled.notna().any():
            ax.plot(filled.index, filled.values, color="tab:red", lw=1.3, label="15-min filled (missing)")
        ax.set_title(f"sensor {sensor_ids[j]} (index {j})")
        ax.set_ylabel("flow")
        ax.legend(loc="upper right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("time")
    save_fig(fig, name)


def plot_missing(summary, missing15, index):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].hist(summary["per_sensor_15"] * 100, bins=50, color="tab:blue")
    axes[0].set_xlabel("missing rate per sensor (%)")
    axes[0].set_ylabel("sensors")
    axes[0].set_title("15-min missing rate per sensor")
    daily = pd.Series(missing15.mean(axis=1), index=index).resample("1D").mean() * 100
    axes[1].plot(daily.index, daily.values, color="tab:red", lw=0.8)
    axes[1].axvline(pd.Timestamp(config.VAL_START), color="k", ls="--", lw=0.8)
    axes[1].axvline(pd.Timestamp(config.TEST_START), color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("missing (%)")
    axes[1].set_title("daily missing rate across sensors")
    save_fig(fig, "q3a_missing")


def plot_overview(flow_frame, mask):
    observed = flow_frame.where(mask)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    daily = observed.mean(axis=1).resample("1D").mean()
    axes[0].plot(daily.index, daily.values, lw=0.8)
    axes[0].axvline(pd.Timestamp(config.VAL_START), color="k", ls="--", lw=0.8, label="val start")
    axes[0].axvline(pd.Timestamp(config.TEST_START), color="r", ls="--", lw=0.8, label="test start")
    axes[0].set_title("daily mean flow across sensors")
    axes[0].set_ylabel("flow")
    axes[0].legend()
    profile = observed.mean(axis=1)
    weekend = profile.index.dayofweek >= 5
    minutes = profile.index.hour * 60 + profile.index.minute
    for flag, label in [(False, "weekday"), (True, "weekend")]:
        curve = profile[weekend == flag].groupby(minutes[weekend == flag]).mean()
        axes[1].plot(curve.index / 60, curve.values, label=label)
    axes[1].set_xlabel("hour of day")
    axes[1].set_title("mean daily profile")
    axes[1].legend()
    save_fig(fig, "q3a_flow_overview")


def main():
    make_dirs()
    set_plot_style()
    log = Logger(config.RESULTS_DIR / "prepare_data.log")
    free = prepare_raw_files()
    log(f"raw files ready, {free / 1024 ** 3:.1f} GB free on disk")
    meta = load_ca_meta()
    sd = select_sd(meta)
    log(f"CA meta: {len(meta)} sensors, San Diego (district {config.SD_DISTRICT}): {len(sd)} sensors")
    if len(sd) != config.EXPECTED_NODES:
        log(f"WARNING: expected {config.EXPECTED_NODES} SD sensors, found {len(sd)}")
    sensor_ids = sd["id"].tolist()

    raw, found = load_raw(sensor_ids, log)
    if not found.all():
        log(f"WARNING: {int((~found).sum())} SD sensors missing from h5 files")
    raw_missing = raw.isna().to_numpy()
    log(f"5-min grid: {raw.shape}, missing {raw_missing.mean() * 100:.2f}%")

    flow15 = resample(raw)
    missing15 = flow15.isna().to_numpy()
    flow, mask = fill_missing(flow15)
    flow_frame = pd.DataFrame(flow, index=flow15.index)
    log(f"15-min grid ({config.RESAMPLE_METHOD}): {flow.shape}, missing {missing15.mean() * 100:.2f}%")

    adj = load_sd_adjacency(sd)
    plot_sensors = choose_plot_sensors(flow, mask)
    log(f"adjacency {adj.shape}, plot sensors {plot_sensors.tolist()}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.PROCESSED_FILE,
        flow=flow,
        mask=mask,
        timestamps=flow15.index.to_numpy().astype("datetime64[ns]"),
        sensor_ids=np.array(sensor_ids),
        adj=adj,
        lat=sd["lat"].to_numpy(dtype=np.float64),
        lng=sd["lng"].to_numpy(dtype=np.float64),
        plot_sensors=plot_sensors,
    )
    sd.to_csv(config.SD_META_FILE, index=False)
    log(f"saved {config.PROCESSED_FILE}")

    summary = missing_summary(raw_missing, missing15, flow_frame)
    observed_values = flow[mask]
    raw_values = raw.to_numpy()
    stats = {
        "num_sensors_ca": len(meta),
        "num_sensors_sd": len(sd),
        "sensors_found_in_h5": int(found.sum()),
        "years": config.YEARS,
        "raw_freq": config.RAW_FREQ,
        "freq": config.FREQ,
        "resample_method": config.RESAMPLE_METHOD,
        "timestamps_5min": raw.shape[0],
        "timestamps_15min": flow.shape[0],
        "time_start": str(flow15.index[0]),
        "time_end": str(flow15.index[-1]),
        "raw_flow_min": float(np.nanmin(raw_values)),
        "raw_flow_max": float(np.nanmax(raw_values)),
        "raw_flow_mean": float(np.nanmean(raw_values)),
        "flow15_mean": float(observed_values.mean()),
        "flow15_std": float(observed_values.std()),
        "flow15_min": float(observed_values.min()),
        "flow15_max": float(observed_values.max()),
        "zero_fraction_observed_15min": float((observed_values == 0).mean()),
        "missing_handling": "resample mean ignores missing 5-min readings; remaining gaps forward-filled "
                            "(causal), leading gaps filled with the sensor's training mean; "
                            "filled points are masked out of loss and metrics",
        "plot_sensors": plot_sensors.tolist(),
        "plot_sensor_ids": [sensor_ids[j] for j in plot_sensors],
        "plot_sensor_mean_flow": [float(flow[mask[:, j], j].mean()) for j in plot_sensors],
        **{k: v for k, v in summary.items() if not k.startswith("per_sensor")},
    }
    save_json(stats, config.RESULTS_DIR / "q3a_data_stats.json")
    np.savez(config.RESULTS_DIR / "q3a_missing_per_sensor.npz", rate_5min=summary["per_sensor_5"],
             rate_15min=summary["per_sensor_15"])

    plot_week(raw, flow_frame, mask, plot_sensors[:3], sensor_ids, config.PLOT_WEEK_START, "q3a_week_three_sensors")
    plot_missing(summary, missing15, flow15.index)
    plot_overview(flow_frame, mask)
    log("Q3(a) done")


if __name__ == "__main__":
    main()
