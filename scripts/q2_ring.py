import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import config
from src.utils import make_dirs, save_fig, save_json, set_plot_style, set_seed

N = 40
STEPS = 20


def behind(x):
    return np.roll(x, 1)


def ahead(x):
    return np.roll(x, -1)


def true_backward(x):
    return np.roll(x, -1)


def true_forward(x):
    return np.roll(x, 1)


def symmetric_layer(x, a, b):
    return a * x + b * (behind(x) + ahead(x))


def directed_layer(x, a, w_minus, w_plus):
    return a * x + w_minus * behind(x) + w_plus * ahead(x)


def history_layer(x, x_prev, a, b, c):
    return a * x + b * (behind(x) + ahead(x)) + c * x_prev


def bump():
    i = np.arange(N)
    return np.exp(-((i - 20) ** 2) / 8)


def spike():
    x = np.zeros(N)
    x[20] = 1.0
    return x


def rollout(step_fn, x0, steps=STEPS):
    xs = [x0]
    for _ in range(steps):
        xs.append(step_fn(xs[-1]))
    return np.stack(xs)


def rollout_history(step_fn, x_prev, x0, steps=STEPS):
    xs = [x_prev, x0]
    for _ in range(steps):
        xs.append(step_fn(xs[-1], xs[-2]))
    return np.stack(xs[1:])


def shape_stats(seq):
    i = np.arange(N)
    rows = []
    for t, x in enumerate(seq):
        total = x.sum()
        angle = np.angle((x * np.exp(2j * np.pi * i / N)).sum())
        centre = round(float(angle * N / (2 * np.pi)), 6) % N
        rows.append({"t": t, "peak": float(x.max()), "sum": float(total), "argmax": int(x.argmax()),
                     "centre_of_mass": float(centre), "nonzero_sensors": int((np.abs(x) > 1e-3).sum())})
    return rows


def part_a_fit():
    x = spike()
    y = true_backward(x)
    features = np.stack([x, behind(x) + ahead(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(features, y, rcond=None)
    pred = symmetric_layer(x, a, b)
    sse = float(((pred - y) ** 2).sum())
    return {
        "a": float(a), "b": float(b), "sse": sse,
        "analytic": {"a": 0.0, "b": 0.5, "sse": 0.5},
        "pred_19_20_21": pred[19:22].tolist(), "true_19_20_21": y[19:22].tolist(),
    }


def heatmaps(sequences, titles, name, vmin=None, vmax=None):
    vmin = min(s.min() for s in sequences) if vmin is None else vmin
    vmax = max(s.max() for s in sequences) if vmax is None else vmax
    fig, axes = plt.subplots(1, len(sequences), figsize=(5 * len(sequences), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, seq, title in zip(axes, sequences, titles):
        image = ax.imshow(seq, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                          extent=[-0.5, N - 0.5, -0.5, len(seq) - 0.5])
        ax.set_title(title)
        ax.set_xlabel("sensor index i")
        ax.set_yticks(range(0, len(seq), 2))
        ax.grid(False)
    axes[0].set_ylabel("time step t")
    fig.colorbar(image, ax=axes.tolist(), label="slowdown x_i(t)")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
    fig.savefig(config.FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def stats_plot(stats, labels, name):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for rows, label in zip(stats, labels):
        t = [r["t"] for r in rows]
        axes[0].plot(t, [r["peak"] for r in rows], marker="o", ms=3, label=label)
        axes[1].plot(t, [r["centre_of_mass"] for r in rows], marker="o", ms=3, label=label)
        axes[2].plot(t, [r["nonzero_sensors"] for r in rows], marker="o", ms=3, label=label)
    axes[0].set_title("peak height")
    axes[1].set_title("centre of mass (sensor)")
    axes[2].set_title("sensors with |x| > 1e-3")
    for ax in axes:
        ax.set_xlabel("time step t")
        ax.legend()
    save_fig(fig, name)


def part_a_rollout(a, b):
    x0 = bump()
    predicted = rollout(lambda x: symmetric_layer(x, a, b), x0)
    true = rollout(true_backward, x0)
    heatmaps([predicted, true], [f"symmetric layer (a={a:.2f}, b={b:.2f})", "true motion x_i(t+1)=x_{i+1}(t)"],
             "q2a_heatmaps")
    stats = [shape_stats(predicted), shape_stats(true)]
    stats_plot(stats, ["symmetric layer", "true motion"], "q2a_bump_stats")
    return {"predicted": stats[0], "true": stats[1],
            "max_abs_error_t20": float(np.abs(predicted[-1] - true[-1]).max())}


def random_snapshots(count, rng):
    return rng.normal(size=(count, N))


def part_b(rng):
    a, w_minus, w_plus = 0.0, 0.0, 1.0
    xs = random_snapshots(200, rng)
    errors = [np.abs(directed_layer(x, a, w_minus, w_plus) - true_backward(x)).max() for x in xs]
    kernel = torch.tensor([[[w_minus, a, w_plus]]], dtype=torch.float64)
    x_t = torch.tensor(xs, dtype=torch.float64)[:, None, :]
    padded = F.pad(x_t, (1, 1), mode="circular")
    conv = F.conv1d(padded, kernel)[:, 0].numpy()
    conv_error = float(np.abs(conv - np.stack([true_backward(x) for x in xs])).max())
    x0 = bump()
    predicted = rollout(lambda x: directed_layer(x, a, w_minus, w_plus), x0)
    true = rollout(true_backward, x0)
    heatmaps([predicted, true], ["directed layer (a=0, w-=0, w+=1)", "true motion"], "q2b_heatmaps")
    return {"a": a, "w_minus": w_minus, "w_plus": w_plus,
            "max_abs_error_random": float(max(errors)),
            "conv_kernel_order_[x_i-1, x_i, x_i+1]": [w_minus, a, w_plus],
            "conv_padding": "circular, 1 on each side",
            "conv_max_abs_error": conv_error,
            "rollout_max_abs_error": float(np.abs(predicted - true).max())}


def fit_history(pairs):
    features, targets = [], []
    for x_prev, x, y in pairs:
        features.append(np.stack([x, behind(x) + ahead(x), x_prev], axis=1))
        targets.append(y)
    coef, *_ = np.linalg.lstsq(np.concatenate(features), np.concatenate(targets), rcond=None)
    return coef


def part_c(rng):
    a, b, c = 0.0, 1.0, -1.0
    backward_pairs, forward_pairs = [], []
    for x_prev in tqdm(random_snapshots(200, rng), desc="Q2(c) random pairs"):
        x = true_backward(x_prev)
        backward_pairs.append((x_prev, x, true_backward(x)))
        x = true_forward(x_prev)
        forward_pairs.append((x_prev, x, true_forward(x)))
    backward_error = max(np.abs(history_layer(x, xp, a, b, c) - y).max() for xp, x, y in backward_pairs)
    forward_error = max(np.abs(history_layer(x, xp, a, b, c) - y).max() for xp, x, y in forward_pairs)

    x0 = bump()
    back_pred = rollout_history(lambda x, xp: history_layer(x, xp, a, b, c), np.roll(x0, 1), x0)
    back_true = rollout(true_backward, x0)
    fwd_pred = rollout_history(lambda x, xp: history_layer(x, xp, a, b, c), np.roll(x0, -1), x0)
    fwd_true = rollout(true_forward, x0)
    heatmaps([back_pred, back_true, fwd_pred, fwd_true],
             ["history layer, backward data", "true backward", "history layer, forward data", "true forward"],
             "q2c_heatmaps")
    return {"a": a, "b": b, "c": c,
            "fitted_backward": fit_history(backward_pairs).tolist(),
            "fitted_forward": fit_history(forward_pairs).tolist(),
            "max_abs_error_backward": float(backward_error),
            "max_abs_error_forward": float(forward_error),
            "rollout_error_backward": float(np.abs(back_pred - back_true).max()),
            "rollout_error_forward": float(np.abs(fwd_pred - fwd_true).max())}


def main():
    set_seed()
    make_dirs()
    set_plot_style()
    rng = np.random.default_rng(config.SEED)
    fit = part_a_fit()
    results = {"part_a_fit": fit, "part_a_rollout": part_a_rollout(fit["a"], fit["b"]),
               "part_b": part_b(rng), "part_c": part_c(rng)}
    save_json(results, config.RESULTS_DIR / "q2_results.json")
    print(f"Q2(a)i a={fit['a']:.4f} b={fit['b']:.4f} SSE={fit['sse']:.4f}")
    p, t = results["part_a_rollout"]["predicted"][-1], results["part_a_rollout"]["true"][-1]
    print(f"Q2(a)ii t=20 predicted peak={p['peak']:.4f} centre={p['centre_of_mass']:.2f} | "
          f"true peak={t['peak']:.4f} centre={t['centre_of_mass']:.2f}")
    print(f"Q2(b) max error={results['part_b']['max_abs_error_random']:.2e} conv error="
          f"{results['part_b']['conv_max_abs_error']:.2e}")
    c = results["part_c"]
    print(f"Q2(c) backward error={c['max_abs_error_backward']:.2e} forward error={c['max_abs_error_forward']:.2e} "
          f"fitted backward={np.round(c['fitted_backward'], 4)} fitted forward={np.round(c['fitted_forward'], 4)}")


if __name__ == "__main__":
    main()
