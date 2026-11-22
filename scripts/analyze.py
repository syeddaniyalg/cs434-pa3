import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

from src import config
from src.data import load_processed, split_ranges, train_correlations
from src.graph import choose_normalization, degrees, haversine_km, hop_matrix, normalize_adjacency, remove_self_loops
from src.utils import Logger, load_json, make_dirs, save_fig, save_json, set_plot_style

MAIN_RUNS = ["temporal", "fixed", "adaptive", "hybrid"]
COLORS = {"temporal": "tab:gray", "fixed": "tab:blue", "adaptive": "tab:orange", "hybrid": "tab:green"}
TOPK = [1, 5, 10, 20]


def color(name):
    return COLORS.get(name, None)


def load_runs():
    runs = {}
    for path in sorted(config.RESULTS_DIR.glob("*/metrics.json")):
        runs[path.parent.name] = load_json(path)
    ordered = [n for n in MAIN_RUNS if n in runs] + sorted(n for n in runs if n not in MAIN_RUNS)
    return {n: runs[n] for n in ordered}


def table_rows(runs):
    rows = []
    for name, m in runs.items():
        row = {
            "Run": name,
            "Model": m["model"],
            "Adjacency": m["adjacency"],
            "Graph layers": m["config"]["graph_layers"],
            "MAE": m["test"]["mae"],
            "RMSE": m["test"]["rmse"],
            "MAPE (%)": m["test"]["mape"],
            "Val MAE": m["val"]["mae"],
            "Val RMSE": m["val"]["rmse"],
            "Training time (s)": m["train_time_sec"],
            "Time/epoch (s)": m["time_per_epoch_sec"],
            "Epochs": m["epochs_run"],
            "Best epoch": m["best_epoch"],
            "Params": m["parameters"],
            "Alpha": m["alpha"],
        }
        for horizon, values in m["test"]["horizons"].items():
            row[f"MAE@{horizon}"] = values["mae"]
            row[f"RMSE@{horizon}"] = values["rmse"]
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def write_markdown(df, path):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.values) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_latex(df, path):
    lines = ["\\begin{tabular}{" + "l" * 2 + "r" * (len(df.columns) - 2) + "}", "\\toprule",
             " & ".join(df.columns) + " \\\\", "\\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(fmt(v) for v in row.values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines).replace("%", "\\%") + "\n")


def save_tables(df, log):
    main = df[df["Run"].isin(MAIN_RUNS)].copy()
    required = main[["Model", "Adjacency", "MAE", "RMSE", "Training time (s)"]]
    required.to_csv(config.TABLES_DIR / "q6a_comparison.csv", index=False)
    write_markdown(required, config.TABLES_DIR / "q6a_comparison.md")
    write_latex(required, config.TABLES_DIR / "q6a_comparison.tex")
    detail_cols = ["Model", "Adjacency", "MAE", "RMSE", "MAPE (%)", "Val MAE", "Val RMSE"] + \
                  [c for c in df.columns if c.startswith("MAE@")] + ["Training time (s)", "Time/epoch (s)",
                                                                     "Epochs", "Best epoch", "Params"]
    write_markdown(main[detail_cols], config.TABLES_DIR / "q6a_comparison_detailed.md")
    write_latex(main[detail_cols], config.TABLES_DIR / "q6a_comparison_detailed.tex")
    main[detail_cols].to_csv(config.TABLES_DIR / "q6a_comparison_detailed.csv", index=False)
    df.to_csv(config.TABLES_DIR / "all_runs.csv", index=False)
    write_markdown(df, config.TABLES_DIR / "all_runs.md")
    for _, row in required.iterrows():
        log(f"{row['Model']:<14} {row['Adjacency']:<9} MAE {row['MAE']:.3f} RMSE {row['RMSE']:.3f} "
            f"time {row['Training time (s)']:.1f}s")


def load_predictions(name):
    path = config.RESULTS_DIR / name / "test_predictions.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def plot_predictions(runs, sensor_ids, start, days, horizons, name):
    preds = {n: load_predictions(n) for n in runs if n in MAIN_RUNS}
    preds = {n: p for n, p in preds.items() if p is not None}
    if not preds:
        return
    first = next(iter(preds.values()))
    nodes = first["nodes"][:3]
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(days=days)
    fig, axes = plt.subplots(len(nodes), len(horizons), figsize=(7 * len(horizons), 2.8 * len(nodes)),
                             sharex=True, squeeze=False)
    for r, node in enumerate(nodes):
        for c, h in enumerate(horizons):
            ax = axes[r, c]
            times = pd.DatetimeIndex(first["target_time"][:, h])
            keep = (times >= start) & (times < end)
            order = np.argsort(times[keep])
            truth = np.where(first["mask"][keep, h, r], first["true"][keep, h, r], np.nan)[order]
            ax.plot(times[keep][order], truth, color="black", lw=1.6, label="actual")
            for run, p in preds.items():
                ax.plot(times[keep][order], p["pred"][keep, h, r][order], color=color(run), lw=1.0, alpha=0.85,
                        label=f"{run}")
            ax.set_title(f"sensor {sensor_ids[node]} | horizon {config.HORIZON_MINUTES[h]} min")
            ax.set_ylabel("flow")
            if r == 0 and c == 0:
                ax.legend(fontsize=8, ncol=5)
    fig.autofmt_xdate()
    save_fig(fig, name)


def prediction_error_table(runs):
    rows = []
    for run in runs:
        p = load_predictions(run)
        if p is None or run not in MAIN_RUNS:
            continue
        for k, node in enumerate(p["nodes"][:3]):
            m = p["mask"][:, :, k]
            err = np.abs(p["pred"][:, :, k] - p["true"][:, :, k])[m]
            rows.append({"run": run, "node": int(node), "mae": float(err.mean())})
    return rows


def plot_training_curves(runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name in runs:
        path = config.RESULTS_DIR / name / "history.csv"
        if not path.exists():
            continue
        h = pd.read_csv(path)
        axes[0].plot(h["epoch"], h["train_mae"], ls="--", color=color(name), alpha=0.7)
        axes[0].plot(h["epoch"], h["val_mae"], color=color(name), label=name)
        axes[1].plot(h["epoch"], h["cumulative_sec"] / 60, color=color(name), label=name)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MAE (flow units)")
    axes[0].set_title("training (dashed) and validation (solid) MAE")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cumulative training time (min)")
    axes[1].set_title("training time")
    axes[1].legend(fontsize=8)
    save_fig(fig, "q6a_training_curves")


def plot_horizons(runs):
    names = [n for n in runs if n in MAIN_RUNS]
    if not names:
        return
    horizons = list(runs[names[0]]["test"]["horizons"].keys())
    width = 0.8 / len(names)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for metric, ax in zip(["mae", "rmse"], axes):
        for i, name in enumerate(names):
            values = [runs[name]["test"]["horizons"][h][metric] for h in horizons]
            ax.bar(np.arange(len(horizons)) + i * width, values, width, label=name, color=color(name))
        ax.set_xticks(np.arange(len(horizons)) + width * (len(names) - 1) / 2)
        ax.set_xticklabels(horizons)
        ax.set_xlabel("forecast horizon")
        ax.set_title(f"test {metric.upper()} by horizon")
        ax.legend(fontsize=8)
    save_fig(fig, "q6a_horizon_metrics")


def load_node_metrics(name):
    path = config.RESULTS_DIR / name / "node_metrics.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def plot_hour_of_day(runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    summary = {}
    for name in runs:
        if name not in MAIN_RUNS:
            continue
        nm = load_node_metrics(name)
        if nm is None:
            continue
        hour = nm["test_hour_mae"]
        summary[name] = hour.mean(axis=1).tolist()
        axes[0].plot(range(24), hour.mean(axis=1), "o-", ms=3, color=color(name), label=name)
    if "temporal" in summary:
        base = np.array(summary["temporal"])
        for name, values in summary.items():
            if name != "temporal":
                axes[1].plot(range(24), 100 * (base - np.array(values)) / base, "o-", ms=3, color=color(name),
                             label=name)
        axes[1].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("hour of day (target time)")
    axes[0].set_ylabel("test MAE")
    axes[0].set_title("test MAE by hour of day (mean over horizons)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("hour of day")
    axes[1].set_ylabel("MAE reduction vs temporal-only (%)")
    axes[1].set_title("where the graph helps during the day")
    axes[1].legend(fontsize=8)
    save_fig(fig, "q6c_hour_of_day_mae")
    return summary


def plot_node_gains(runs, adj):
    base = load_node_metrics("temporal") if "temporal" in runs else None
    if base is None:
        return {}
    out_degree, _ = degrees(adj)
    names = [n for n in ["fixed", "adaptive", "hybrid"] if n in runs and load_node_metrics(n) is not None]
    if not names:
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    summary = {}
    valid = base["test_node_count"] > 0
    for name in names:
        nm = load_node_metrics(name)
        gain = (base["test_node_mae"] - nm["test_node_mae"])[valid]
        summary[name] = {"nodes_improved_fraction": float((gain > 0).mean()), "mean_gain": float(gain.mean()),
                         "median_gain": float(np.median(gain)),
                         "gain_vs_degree_spearman": float(spearmanr(out_degree[valid], gain)[0])}
        axes[0].hist(gain, bins=60, alpha=0.5, color=color(name), label=name)
        axes[1].scatter(out_degree[valid], gain, s=6, alpha=0.5, color=color(name), label=name)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("per-sensor MAE reduction vs temporal-only")
    axes[0].set_ylabel("sensors")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("road-graph out-degree")
    axes[1].set_ylabel("MAE reduction")
    axes[1].legend(fontsize=8)
    save_fig(fig, "q6a_node_gains")
    return summary


def plot_alpha(runs):
    names = [n for n in runs if runs[n]["model_type"] == "hybrid"]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name in names:
        h = pd.read_csv(config.RESULTS_DIR / name / "history.csv")
        if h["alpha"].notna().any():
            ax.plot(h["epoch"], h["alpha"], "o-", ms=3, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("alpha (weight on fixed graph)")
    ax.set_title("hybrid mixing weight")
    ax.legend(fontsize=8)
    save_fig(fig, "q6b_hybrid_alpha")


def top_neighbours(matrix, k):
    scores = np.array(matrix, dtype=np.float64)
    np.fill_diagonal(scores, -np.inf)
    return np.argsort(-scores, axis=1)[:, :k]


def analyze_learned(learned, physical, hops, distance, corr):
    n = len(learned)
    off = ~np.eye(n, dtype=bool)
    result = {"self_weight_mean": float(np.diag(learned).mean()),
              "row_entropy_mean": float(-(learned * np.log(learned + 1e-12)).sum(1).mean()),
              "max_entropy": float(np.log(n))}
    result["effective_neighbours_mean"] = float(np.exp(result["row_entropy_mean"]))
    corr_top = top_neighbours(corr, 10)
    for k in TOPK:
        idx = top_neighbours(learned, k)
        rows = np.repeat(np.arange(n), k)
        cols = idx.ravel()
        result[f"top{k}"] = {
            "fraction_physical_edge": float(physical[rows, cols].mean()),
            "fraction_within_2_hops": float((hops[rows, cols] <= 2).mean()),
            "fraction_beyond_2_hops": float((hops[rows, cols] > 2).mean()),
            "fraction_unreachable": float(np.isinf(hops[rows, cols]).mean()),
            "mean_distance_km": float(distance[rows, cols].mean()),
            "median_distance_km": float(np.median(distance[rows, cols])),
            "mean_train_corr": float(corr[rows, cols].mean()),
            "weight_share": float(learned[rows, cols].sum() / n),
        }
    idx10 = top_neighbours(learned, 10)
    result["top10_overlap_with_top10_correlated"] = float(np.mean(
        [len(set(idx10[i]) & set(corr_top[i])) / 10 for i in range(n)]))
    rows10 = np.repeat(np.arange(n), 10)
    cols10 = idx10.ravel()
    far = hops[rows10, cols10] > 2
    result["top10_non_local_edges"] = {
        "count": int(far.sum()),
        "mean_train_corr": float(corr[rows10, cols10][far].mean()) if far.any() else None,
        "mean_distance_km": float(distance[rows10, cols10][far].mean()) if far.any() else None,
    }
    values = learned[off]
    strongest = np.argsort(-values)[: n * 5]
    off_rows, off_cols = np.where(off)
    result["global_strongest_5n_edges"] = {
        "fraction_physical_edge": float(physical[off_rows[strongest], off_cols[strongest]].mean()),
        "fraction_within_2_hops": float((hops[off_rows[strongest], off_cols[strongest]] <= 2).mean()),
        "mean_distance_km": float(distance[off_rows[strongest], off_cols[strongest]].mean()),
    }
    result["spearman_weight_vs_distance"] = float(spearmanr(values, distance[off])[0])
    result["spearman_weight_vs_train_corr"] = float(spearmanr(values, corr[off])[0])
    return result


def baselines(physical, hops, distance, corr):
    n = len(physical)
    off = ~np.eye(n, dtype=bool)
    return {
        "random_pair": {"fraction_physical_edge": float(physical[off].mean()),
                        "fraction_within_2_hops": float((hops[off] <= 2).mean()),
                        "mean_distance_km": float(distance[off].mean()),
                        "mean_train_corr": float(corr[off].mean())},
        "physical_edge": {"mean_distance_km": float(distance[physical].mean()) if physical.any() else None,
                          "mean_train_corr": float(corr[physical].mean()) if physical.any() else None},
        "top10_correlated": {
            "fraction_physical_edge": float(physical[np.repeat(np.arange(n), 10),
                                                     top_neighbours(corr, 10).ravel()].mean()),
            "mean_distance_km": float(distance[np.repeat(np.arange(n), 10), top_neighbours(corr, 10).ravel()].mean()),
        },
    }


def plot_adjacency_comparison(matrices):
    names = list(matrices.keys())
    fig, axes = plt.subplots(2, len(names), figsize=(5.2 * len(names), 10), squeeze=False)
    for c, name in enumerate(names):
        m = np.array(matrices[name], dtype=np.float64)
        positive = m[m > 0]
        vmin = max(np.percentile(positive, 50), 1e-6) if positive.size else 1e-6
        norm = LogNorm(vmin=vmin, vmax=max(m.max(), vmin * 10))
        image = axes[0, c].imshow(np.clip(m, vmin, None), cmap="magma", norm=norm, interpolation="nearest")
        axes[0, c].set_title(f"{name} (full)")
        fig.colorbar(image, ax=axes[0, c], fraction=0.046)
        block = slice(0, min(80, len(m)))
        image = axes[1, c].imshow(m[block, block], cmap="magma", interpolation="nearest")
        axes[1, c].set_title(f"{name} (first 80 x 80)")
        fig.colorbar(image, ax=axes[1, c], fraction=0.046)
        for ax in axes[:, c]:
            ax.set_xlabel("sensor j")
            ax.set_ylabel("sensor i")
            ax.grid(False)
    save_fig(fig, "q5_adjacency_comparison")


def plot_overlap(analysis, base):
    names = list(analysis.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        a = analysis[name]
        axes[0].bar(np.arange(len(TOPK)) + i * width, [a[f"top{k}"]["fraction_physical_edge"] for k in TOPK], width,
                    label=name)
        axes[1].bar(np.arange(len(TOPK)) + i * width, [a[f"top{k}"]["mean_distance_km"] for k in TOPK], width,
                    label=name)
        axes[2].bar(np.arange(len(TOPK)) + i * width, [a[f"top{k}"]["mean_train_corr"] for k in TOPK], width,
                    label=name)
    axes[0].axhline(base["random_pair"]["fraction_physical_edge"], color="k", ls="--", label="random pair")
    axes[1].axhline(base["random_pair"]["mean_distance_km"], color="k", ls="--", label="random pair")
    if base["physical_edge"]["mean_distance_km"] is not None:
        axes[1].axhline(base["physical_edge"]["mean_distance_km"], color="tab:blue", ls=":", label="road edge")
        axes[2].axhline(base["physical_edge"]["mean_train_corr"], color="tab:blue", ls=":", label="road edge")
    axes[2].axhline(base["random_pair"]["mean_train_corr"], color="k", ls="--", label="random pair")
    titles = ["fraction of learned top-k edges in road graph", "mean distance of top-k learned neighbours (km)",
              "mean local correlation of top-k learned neighbours"]
    for ax, title in zip(axes, titles):
        ax.set_xticks(np.arange(len(TOPK)) + width * (len(names) - 1) / 2)
        ax.set_xticklabels([f"top-{k}" for k in TOPK])
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
    save_fig(fig, "q6b_learned_vs_physical")


def plot_distance_corr(learned, physical, distance, corr, name):
    n = len(learned)
    off = ~np.eye(n, dtype=bool)
    idx = top_neighbours(learned, 10)
    rows, cols = np.repeat(np.arange(n), 10), idx.ravel()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    bins = np.linspace(0, np.percentile(distance[off], 99), 60)
    axes[0].hist(distance[off], bins=bins, density=True, alpha=0.4, color="gray", label="random pairs")
    if physical.any():
        axes[0].hist(distance[physical], bins=bins, density=True, alpha=0.5, color="tab:blue", label="road edges")
    axes[0].hist(distance[rows, cols], bins=bins, density=True, alpha=0.5, color="tab:orange", label="learned top-10")
    axes[0].set_xlabel("distance (km)")
    axes[0].set_title("geographic distance")
    axes[0].legend(fontsize=8)
    cbins = np.linspace(-0.2, 1, 60)
    axes[1].hist(corr[off], bins=cbins, density=True, alpha=0.4, color="gray", label="random pairs")
    if physical.any():
        axes[1].hist(corr[physical], bins=cbins, density=True, alpha=0.5, color="tab:blue", label="road edges")
    axes[1].hist(corr[rows, cols], bins=cbins, density=True, alpha=0.5, color="tab:orange", label="learned top-10")
    axes[1].set_xlabel("local correlation (train)")
    axes[1].set_title("traffic correlation")
    axes[1].legend(fontsize=8)
    sample = np.random.default_rng(config.SEED).choice(off.sum(), size=min(40000, off.sum()), replace=False)
    axes[2].hexbin(corr[off][sample], learned[off][sample], gridsize=50, bins="log", cmap="viridis", yscale="log")
    axes[2].set_xlabel("local correlation (train)")
    axes[2].set_ylabel("learned weight")
    axes[2].set_title("learned weight vs correlation")
    fig.suptitle(name)
    save_fig(fig, f"q6b_distance_correlation_{name.replace(' ', '_').replace('(', '').replace(')', '')}")


def plot_learned_map(learned, physical, lat, lng, name):
    idx = top_neighbours(learned, 1)[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    rows, cols = np.where(physical)
    axes[0].add_collection(LineCollection([[(lng[i], lat[i]), (lng[j], lat[j])] for i, j in zip(rows, cols)],
                                          colors="tab:blue", linewidths=0.4, alpha=0.6))
    axes[0].scatter(lng, lat, s=5, color="black")
    axes[0].set_title("road-distance graph")
    local = [[(lng[i], lat[i]), (lng[j], lat[j])] for i, j in enumerate(idx) if physical[i, j]]
    remote = [[(lng[i], lat[i]), (lng[j], lat[j])] for i, j in enumerate(idx) if not physical[i, j]]
    axes[1].add_collection(LineCollection(remote, colors="tab:red", linewidths=0.5, alpha=0.5,
                                          label=f"not a road edge ({len(remote)})"))
    axes[1].add_collection(LineCollection(local, colors="tab:green", linewidths=0.9, alpha=0.9,
                                          label=f"road edge ({len(local)})"))
    axes[1].scatter(lng, lat, s=5, color="black")
    axes[1].set_title(f"{name}: strongest learned neighbour of each sensor")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.autoscale()
    save_fig(fig, f"q6b_learned_map_{name.replace(' ', '_').replace('(', '').replace(')', '')}")


def graph_analysis(runs, processed, log):
    adj = processed["adj"]
    lat, lng = processed["lat"], processed["lng"]
    physical = remove_self_loops(adj) != 0
    physical = physical | physical.T
    hops = hop_matrix(adj)
    hops = np.minimum(hops, hops.T)
    distance = haversine_km(lat, lng)
    corr = train_correlations(processed["flow"], processed["mask"], processed["timestamps"])[1]
    fixed_norm = normalize_adjacency(adj, choose_normalization(adj))
    learned = {}
    if "adaptive" in runs and (config.RESULTS_DIR / "adaptive" / "adjacency.npy").exists():
        learned["adaptive"] = np.load(config.RESULTS_DIR / "adaptive" / "adjacency.npy")
    if "hybrid" in runs and (config.RESULTS_DIR / "hybrid" / "adaptive_component.npy").exists():
        learned["hybrid (adaptive part)"] = np.load(config.RESULTS_DIR / "hybrid" / "adaptive_component.npy")
        learned["hybrid (combined)"] = np.load(config.RESULTS_DIR / "hybrid" / "adjacency.npy")
    if not learned:
        log("no learned adjacency found, skipping graph analysis")
        return {}
    plot_adjacency_comparison({"fixed (normalized)": fixed_norm, **learned})
    train = slice(*split_ranges(processed["timestamps"])["train"])
    keep = np.where(processed["mask"][train].any(axis=0))[0]
    sub = np.ix_(keep, keep)
    physical, hops, distance, corr, fixed_norm = physical[sub], hops[sub], distance[sub], corr[sub], fixed_norm[sub]
    learned = {name: m[sub] for name, m in learned.items()}
    lat, lng = lat[keep], lng[keep]
    log(f"graph analysis on {len(keep)} sensors observed in training ({len(adj) - len(keep)} never-observed excluded)")
    base = baselines(physical, hops, distance, corr)
    analysis = {name: analyze_learned(m, physical, hops, distance, corr) for name, m in learned.items()}
    analysis["fixed (normalized)"] = analyze_learned(fixed_norm, physical, hops, distance, corr)
    plot_overlap(analysis, base)
    for name, m in learned.items():
        plot_distance_corr(m, physical, distance, corr, name)
        plot_learned_map(m, physical, lat, lng, name)
        a = analysis[name]
        log(f"{name}: top-10 in road graph {a['top10']['fraction_physical_edge']:.3f} "
            f"(random {base['random_pair']['fraction_physical_edge']:.3f}), within 2 hops "
            f"{a['top10']['fraction_within_2_hops']:.3f}, mean km {a['top10']['mean_distance_km']:.1f} "
            f"(random {base['random_pair']['mean_distance_km']:.1f}), mean corr {a['top10']['mean_train_corr']:.3f} "
            f"(random {base['random_pair']['mean_train_corr']:.3f})")
    return {"analysis_nodes": len(keep), "excluded_never_observed": len(adj) - len(keep),
            "baselines": base, "learned": analysis}


def relative_gains(runs):
    if "temporal" not in runs:
        return {}
    base = runs["temporal"]["test"]
    gains = {}
    for name, m in runs.items():
        if name == "temporal":
            continue
        gains[name] = {"mae_reduction_pct": 100 * (base["mae"] - m["test"]["mae"]) / base["mae"],
                       "rmse_reduction_pct": 100 * (base["rmse"] - m["test"]["rmse"]) / base["rmse"]}
    return gains


def main():
    make_dirs()
    set_plot_style()
    log = Logger(config.RESULTS_DIR / "analyze.log")
    runs = load_runs()
    if not runs:
        log("no runs found in outputs/results, train models first")
        return
    log(f"found runs: {list(runs.keys())}")
    processed = load_processed()
    df = table_rows(runs)
    save_tables(df, log)
    sensor_ids = processed["sensor_ids"]
    plot_predictions(runs, sensor_ids, config.TEST_PLOT_WEEK_START, 7, [0, 3], "q6a_pred_vs_actual_week")
    plot_predictions(runs, sensor_ids, config.TEST_PLOT_WEEK_START, 1, [0, 3], "q6a_pred_vs_actual_day")
    plot_training_curves(runs)
    plot_horizons(runs)
    hour_summary = plot_hour_of_day(runs)
    node_summary = plot_node_gains(runs, processed["adj"])
    plot_alpha(runs)
    graph = graph_analysis(runs, processed, log)
    summary = {
        "comparison": df.to_dict(orient="records"),
        "relative_to_temporal": relative_gains(runs),
        "plot_sensor_mae": prediction_error_table(runs),
        "hour_of_day_test_mae": hour_summary,
        "node_gains_vs_temporal": node_summary,
        "graph_analysis": graph,
    }
    save_json(summary, config.RESULTS_DIR / "q6_analysis.json")
    log(f"relative gains vs temporal-only: {summary['relative_to_temporal']}")
    log("Q6 analysis done")


if __name__ == "__main__":
    main()
