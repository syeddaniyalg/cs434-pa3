import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


def remove_self_loops(adj):
    adj = np.array(adj, dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def is_symmetric(adj, tol=1e-6):
    return bool(np.allclose(adj, adj.T, atol=tol))


def choose_normalization(adj):
    return "sym" if is_symmetric(remove_self_loops(adj)) else "row"


def normalize_adjacency(adj, mode="sym"):
    a = remove_self_loops(adj) + np.eye(len(adj), dtype=np.float32)
    degree = a.sum(axis=1)
    if mode == "sym":
        inv_sqrt = 1.0 / np.sqrt(degree)
        return (inv_sqrt[:, None] * a * inv_sqrt[None, :]).astype(np.float32)
    if mode == "row":
        return (a / degree[:, None]).astype(np.float32)
    raise ValueError(f"unknown normalization {mode}")


def edge_stats(adj):
    a = remove_self_loops(adj)
    binary = a != 0
    out_degree = binary.sum(axis=1)
    in_degree = binary.sum(axis=0)
    undirected = binary | binary.T
    weights = a[binary]
    reciprocal = (binary & binary.T).sum()
    weak = connected_components(csr_matrix(binary), directed=True, connection="weak")[0]
    strong = connected_components(csr_matrix(binary), directed=True, connection="strong")[0]
    return {
        "num_nodes": len(a),
        "nonzero_entries_total": int((np.asarray(adj) != 0).sum()),
        "nonzero_diagonal": int((np.diag(adj) != 0).sum()),
        "nonzero_edges_no_self_loops": int(binary.sum()),
        "undirected_edges": int(np.triu(undirected, 1).sum()),
        "density": float(binary.sum() / (len(a) * (len(a) - 1))),
        "symmetric": is_symmetric(a),
        "symmetric_pattern": bool((binary == binary.T).all()),
        "reciprocal_edge_fraction": float(reciprocal / max(binary.sum(), 1)),
        "max_abs_asymmetry": float(np.abs(a - a.T).max()),
        "weight_min": float(weights.min()) if weights.size else 0.0,
        "weight_max": float(weights.max()) if weights.size else 0.0,
        "weight_mean": float(weights.mean()) if weights.size else 0.0,
        "out_degree_mean": float(out_degree.mean()),
        "out_degree_median": float(np.median(out_degree)),
        "out_degree_max": int(out_degree.max()),
        "out_degree_min": int(out_degree.min()),
        "in_degree_mean": float(in_degree.mean()),
        "isolated_nodes": int(((out_degree == 0) & (in_degree == 0)).sum()),
        "weak_components": int(weak),
        "strong_components": int(strong),
    }


def degrees(adj):
    binary = remove_self_loops(adj) != 0
    return binary.sum(axis=1), binary.sum(axis=0)


def hop_matrix(adj):
    binary = csr_matrix((remove_self_loops(adj) != 0).astype(np.float32))
    return shortest_path(binary, directed=True, unweighted=True)


def k_hop_sets(adj, node, k=2):
    binary = remove_self_loops(adj) != 0
    seen = {node}
    frontier = {node}
    layers = []
    for _ in range(k):
        nxt = set()
        for u in frontier:
            nxt.update(np.where(binary[u])[0].tolist())
        nxt -= seen
        layers.append(sorted(nxt))
        seen |= nxt
        frontier = nxt
    return layers


def reachable_fraction(hops, max_k=10):
    n = len(hops)
    rows = []
    for k in range(1, max_k + 1):
        within = ((hops <= k) & (hops > 0)).sum(axis=1)
        rows.append(float(within.mean() / (n - 1)))
    return rows


def haversine_km(lat, lng):
    lat = np.radians(lat)[:, None]
    lng = np.radians(lng)[:, None]
    dlat = lat - lat.T
    dlng = lng - lng.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlng / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
