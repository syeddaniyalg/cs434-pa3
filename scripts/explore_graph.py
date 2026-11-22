import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from tqdm import tqdm

from src import config
from src.data import load_processed, split_ranges, train_correlations
from src.graph import (choose_normalization, degrees, edge_stats, haversine_km, hop_matrix, k_hop_sets,
                       normalize_adjacency, reachable_fraction, remove_self_loops)
from src.utils import Logger, make_dirs, save_fig, save_json, set_plot_style

MAX_HOP = 10
SMOOTH_STEPS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def plot_adjacency(adj):
    a = remove_self_loops(adj)
    positive = a[a > 0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    image = axes[0].imshow(np.where(a > 0, a, np.nan), cmap="viridis", interpolation="nearest",
                           norm=LogNorm(vmin=positive.min(), vmax=positive.max()) if positive.size else None)
    axes[0].set_title("SD road-distance adjacency (log colour)")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].spy(a != 0, markersize=0.6)
    axes[1].set_title("nonzero pattern")
    block = slice(0, min(100, len(a)))
    image = axes[2].imshow(a[block, block], cmap="viridis", interpolation="nearest")
    axes[2].set_title("first 100 x 100 block")
    fig.colorbar(image, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("sensor j")
        ax.set_ylabel("sensor i")
        ax.grid(False)
    save_fig(fig, "q3b_adjacency_heatmap")


def plot_degrees(adj):
    out_degree, in_degree = degrees(adj)
    a = remove_self_loops(adj)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    bins = np.arange(0, max(out_degree.max(), in_degree.max()) + 2) - 0.5
    axes[0].hist(out_degree, bins=bins, color="tab:blue")
    axes[0].set_title(f"out-degree (mean {out_degree.mean():.2f})")
    axes[0].set_xlabel("neighbours |N(i)|")
    axes[0].set_ylabel("sensors")
    axes[1].hist(in_degree, bins=bins, color="tab:orange")
    axes[1].set_title(f"in-degree (mean {in_degree.mean():.2f})")
    axes[1].set_xlabel("incoming neighbours")
    axes[2].hist(a[a > 0], bins=50, color="tab:green")
    axes[2].set_title("edge weight distribution")
    axes[2].set_xlabel("A_ij")
    save_fig(fig, "q3b_degree_distribution")


def pick_sensor(adj):
    out_degree, _ = degrees(adj)
    target = np.median(out_degree[out_degree > 0])
    order = np.argsort(np.abs(out_degree - target) + np.arange(len(adj)) * 1e-6)
    for node in order:
        one, two = k_hop_sets(adj, int(node), 2)
        if len(one) > 0 and len(two) > 0:
            return int(node)
    return int(order[0])


def plot_neighbourhood(node, one, two, lat, lng, adj, sensor_ids):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(lng, lat, s=6, color="lightgray", label="other sensors")
    ax.scatter(lng[two], lat[two], s=30, color="tab:orange", label=f"2-hop ({len(two)})")
    ax.scatter(lng[one], lat[one], s=40, color="tab:red", label=f"1-hop ({len(one)})")
    for j in one:
        ax.plot([lng[node], lng[j]], [lat[node], lat[j]], color="tab:red", lw=0.8)
        for k in np.where(remove_self_loops(adj)[j] != 0)[0]:
            if k in two:
                ax.plot([lng[j], lng[k]], [lat[j], lat[k]], color="tab:orange", lw=0.6, alpha=0.7)
    ax.scatter(lng[node], lat[node], s=200, marker="*", color="black", label=f"sensor {sensor_ids[node]}")
    pad = 0.05
    close = [node] + one + two
    ax.set_xlim(lng[close].min() - pad, lng[close].max() + pad)
    ax.set_ylim(lat[close].min() - pad, lat[close].max() + pad)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("one-hop and two-hop neighbourhood")
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, "q3c_neighbourhood_map")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(lng, lat, s=4, color="tab:blue")
    a = remove_self_loops(adj)
    rows, cols = np.where(a != 0)
    segments = [[(lng[i], lat[i]), (lng[j], lat[j])] for i, j in zip(rows, cols)]
    ax.add_collection(LineCollection(segments, colors="gray", linewidths=0.3, alpha=0.6))
    ax.scatter(lng[node], lat[node], s=150, marker="*", color="red")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("SD sensors and road-distance edges")
    save_fig(fig, "q3b_graph_map")


def undirected_hops(hops):
    return np.minimum(hops, hops.T)


def correlation_by_hop(corr, hops):
    groups = {}
    n = len(corr)
    off = ~np.eye(n, dtype=bool)
    for h in range(1, MAX_HOP + 1):
        values = corr[(hops == h) & off]
        if values.size:
            groups[str(h)] = values
    far = corr[(hops > MAX_HOP) & np.isfinite(hops) & off]
    if far.size:
        groups[f">{MAX_HOP}"] = far
    unreachable = corr[np.isinf(hops) & off]
    if unreachable.size:
        groups["unreachable"] = unreachable
    return groups


def plot_correlation(panels, node_hops, node):
    fig, axes = plt.subplots(len(panels), 2, figsize=(14, 4 * len(panels)), squeeze=False)
    for row, (title, groups, node_corr) in enumerate(panels):
        labels = list(groups.keys())
        axes[row, 0].boxplot([groups[k] for k in labels], tick_labels=labels, showfliers=False)
        axes[row, 0].plot(range(1, len(labels) + 1), [groups[k].mean() for k in labels], "o-", color="tab:red",
                          label="mean")
        axes[row, 0].set_xlabel("hop distance (undirected)")
        axes[row, 0].tick_params(axis="x", labelrotation=45)
        axes[row, 0].set_ylabel(f"{title} correlation")
        axes[row, 0].set_title(f"all sensor pairs: {title} correlation vs hop distance")
        axes[row, 0].legend()
        buckets = {"1-hop": node_hops == 1, "2-hop": node_hops == 2,
                   "3-5 hops": (node_hops >= 3) & (node_hops <= 5),
                   "6+ hops": (node_hops >= 6) & np.isfinite(node_hops), "unreachable": np.isinf(node_hops)}
        buckets = {k: node_corr[v] for k, v in buckets.items() if v.any()}
        axes[row, 1].boxplot(list(buckets.values()), tick_labels=list(buckets.keys()), showfliers=False)
        axes[row, 1].set_title(f"selected sensor {node}: {title} correlation with others")
        axes[row, 1].set_ylabel(f"{title} correlation")
    save_fig(fig, "q3c_correlation_by_hop")


def oversmoothing(adj, flow, mask, ranges):
    norm = choose_normalization(adj)
    a_hat = normalize_adjacency(adj, norm).astype(np.float64)
    train = flow[slice(*ranges["train"])]
    alive = mask[slice(*ranges["train"])].any(axis=0)
    x = (train - train.mean(0)) / (train.std(0) + 1e-6)
    x = x[: config.STEPS_PER_DAY * 7].T.astype(np.float64)
    rows = []
    h = x.copy()
    for k in tqdm(range(max(SMOOTH_STEPS) + 1), desc="oversmoothing propagation", leave=False):
        if k > 0:
            h = a_hat @ h
        if k in SMOOTH_STEPS:
            centred = h - h.mean(axis=1, keepdims=True)
            std = centred.std(axis=1)
            valid = (std > 1e-8) & alive
            corr = np.corrcoef(h[valid])
            off = ~np.eye(len(corr), dtype=bool)
            self_corr = np.array([np.corrcoef(h[i], x[i])[0, 1] for i in np.where(valid)[0]])
            node_spread = h[alive].std(axis=0).mean()
            rows.append({"layers": k, "mean_pairwise_corr": float(np.nanmean(corr[off])),
                         "mean_corr_with_own_input": float(np.nanmean(self_corr)),
                         "across_node_std": float(node_spread)})
    return rows, norm


def plot_oversmoothing(rows, reach):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    k = [r["layers"] for r in rows]
    axes[0].plot(k, [r["mean_pairwise_corr"] for r in rows], "o-", label="mean pairwise corr between sensors")
    axes[0].plot(k, [r["mean_corr_with_own_input"] for r in rows], "s-", label="corr with own original series")
    axes[0].plot(k, np.array([r["across_node_std"] for r in rows]) / rows[0]["across_node_std"], "^-",
                 label="across-sensor std (relative)")
    axes[0].set_xlabel("propagation steps k (A_hat^k X)")
    axes[0].set_title("repeated graph propagation smooths sensors")
    axes[0].legend(fontsize=8)
    axes[1].plot(range(1, len(reach) + 1), np.array(reach) * 100, "o-")
    axes[1].set_xlabel("hops k")
    axes[1].set_ylabel("reachable sensors (%)")
    axes[1].set_title("receptive field growth")
    save_fig(fig, "q3c_oversmoothing")


def main():
    np.seterr(invalid="ignore", divide="ignore")
    make_dirs()
    set_plot_style()
    log = Logger(config.RESULTS_DIR / "explore_graph.log")
    data = load_processed()
    adj = data["adj"]
    sensor_ids = data["sensor_ids"]
    lat, lng = data["lat"], data["lng"]
    ranges = split_ranges(data["timestamps"])
    assert adj.shape == (len(sensor_ids), len(sensor_ids)) == (data["flow"].shape[1],) * 2

    stats = edge_stats(adj)
    stats["sensor_order"] = "adjacency rows/cols and flow columns both follow sd_meta.csv row order (ca_meta District 11 rows)"
    stats["chosen_normalization"] = choose_normalization(adj)
    log(f"Q3(b) nodes {stats['num_nodes']}, nonzero edges {stats['nonzero_edges_no_self_loops']} "
        f"(total nonzero incl. diagonal {stats['nonzero_entries_total']}), symmetric {stats['symmetric']}, "
        f"symmetric pattern {stats['symmetric_pattern']}, normalization {stats['chosen_normalization']}")
    plot_adjacency(adj)
    plot_degrees(adj)

    hops = hop_matrix(adj)
    hops_u = undirected_hops(hops)
    distance = haversine_km(lat, lng)
    a = remove_self_loops(adj)
    edges = a != 0
    stats["edge_distance_km_mean"] = float(distance[edges].mean()) if edges.any() else 0.0
    stats["edge_distance_km_median"] = float(np.median(distance[edges])) if edges.any() else 0.0
    stats["weight_vs_distance_corr"] = float(np.corrcoef(a[edges], distance[edges])[0, 1]) if edges.sum() > 2 else 0.0
    finite = np.isfinite(hops) & (hops > 0)
    stats["mean_shortest_path_hops"] = float(hops[finite].mean()) if finite.any() else 0.0
    stats["diameter_hops"] = float(hops[finite].max()) if finite.any() else 0.0

    node = pick_sensor(adj)
    one, two = k_hop_sets(adj, node, 2)
    neighbour_rows = [{"index": j, "sensor_id": str(sensor_ids[j]), "hop": 1, "weight": float(a[node, j]),
                       "distance_km": float(distance[node, j])} for j in one]
    neighbour_rows += [{"index": j, "sensor_id": str(sensor_ids[j]), "hop": 2,
                        "distance_km": float(distance[node, j])} for j in two]
    log(f"Q3(c) sensor {sensor_ids[node]} (index {node}): {len(one)} one-hop, {len(two)} new two-hop neighbours")
    plot_neighbourhood(node, one, two, lat, lng, adj, sensor_ids)

    corr, corr_res = train_correlations(data["flow"], data["mask"], data["timestamps"])
    groups = correlation_by_hop(corr, hops_u)
    groups_res = correlation_by_hop(corr_res, hops_u)
    plot_correlation([("raw", groups, np.delete(corr[node], node)),
                      ("local", groups_res, np.delete(corr_res[node], node))],
                     np.delete(hops_u[node], node), node)
    reach = reachable_fraction(hops, MAX_HOP)
    smooth_rows, norm = oversmoothing(adj, data["flow"], data["mask"], ranges)
    plot_oversmoothing(smooth_rows, reach)

    result = {
        "q3b": stats,
        "q3c": {
            "selected_sensor_index": node,
            "selected_sensor_id": str(sensor_ids[node]),
            "one_hop_count": len(one),
            "two_hop_new_count": len(two),
            "within_two_hops_count": len(one) + len(two),
            "neighbours": neighbour_rows,
            "selected_corr_mean_one_hop": float(corr[node, one].mean()) if one else None,
            "selected_corr_mean_two_hop": float(corr[node, two].mean()) if two else None,
            "selected_corr_mean_all_others": float(np.delete(corr[node], node).mean()),
            "selected_local_corr_mean_one_hop": float(corr_res[node, one].mean()) if one else None,
            "selected_local_corr_mean_two_hop": float(corr_res[node, two].mean()) if two else None,
            "selected_local_corr_mean_all_others": float(np.delete(corr_res[node], node).mean()),
            "corr_by_hop_mean": {k: float(v.mean()) for k, v in groups.items()},
            "local_corr_by_hop_mean": {k: float(v.mean()) for k, v in groups_res.items()},
            "correlation_note": "local = train series minus each sensor's time-of-week mean profile and minus the network-wide mean at each time step",
            "corr_by_hop_count": {k: int(v.size) for k, v in groups.items()},
            "reachable_fraction_by_hops": reach,
            "oversmoothing_normalization": norm,
            "oversmoothing": smooth_rows,
        },
    }
    save_json(result, config.RESULTS_DIR / "q3bc_graph_stats.json")
    pd.DataFrame(neighbour_rows).to_csv(config.TABLES_DIR / "q3c_neighbours.csv", index=False)
    log("Q3(b,c) done")


if __name__ == "__main__":
    main()
