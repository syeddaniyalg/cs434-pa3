import argparse
import shutil

import numpy as np
import torch

from src import config
from src.data import TrafficData
from src.metrics import scalar_metrics
from src.models import build_model, count_parameters
from src.train import alpha_value, evaluate, train_model, write_history
from src.utils import Logger, get_device, hardware_info, make_dirs, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=config.MODEL_TYPES)
    parser.add_argument("--name", default=None)
    parser.add_argument("--fresh", action="store_true")
    for key, value in config.TRAIN_DEFAULTS.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    return parser.parse_args()


def run_name(args):
    if args.name:
        return args.name
    name = args.model
    if args.graph_layers != 1:
        name += f"_L{args.graph_layers}"
    if args.topk and args.model in ("adaptive", "hybrid"):
        name += f"_top{args.topk}"
    if args.model == "hybrid" and args.alpha != "learn":
        name += f"_alpha{args.alpha}"
    return name


def main():
    args = parse_args()
    cfg = {key: getattr(args, key) for key in config.TRAIN_DEFAULTS}
    name = run_name(args)
    make_dirs()
    run_dir = config.CHECKPOINTS_DIR / name
    out_dir = config.RESULTS_DIR / name
    if args.fresh:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train.log")

    set_seed(config.SEED)
    device = get_device()
    data = TrafficData(device)
    set_seed(config.SEED)
    model, norm = build_model(args.model, data.adj, data.num_nodes, config.INPUT_LEN, config.OUTPUT_LEN, cfg,
                              source_mask=data.alive)
    model = model.to(device)
    hardware = hardware_info()
    log(f"run {name} | model {args.model} | normalization {norm} | params {count_parameters(model)} | device {device}")
    log(f"config {cfg}")
    log(f"data {data.summary()}")
    log(f"hardware {hardware}")

    state = train_model(model, data, cfg, run_dir, log)
    write_history(state["history"], out_dir / "history.csv")

    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    log(f"loaded best model from epoch {best['epoch']} (val MAE {best['val']:.3f})")
    val, _ = evaluate(model, data, "val", cfg["eval_batch_size"], desc="final val")
    test, kept = evaluate(model, data, "test", cfg["eval_batch_size"], keep_nodes=torch.tensor(data.plot_sensors, device=device),
                          desc="final test")

    history = state["history"]
    model_label, adjacency_label = config.MODEL_LABELS[args.model]
    metrics = {
        "name": name,
        "model_type": args.model,
        "model": model_label,
        "adjacency": adjacency_label,
        "normalization": norm,
        "config": cfg,
        "seed": config.SEED,
        "input_len": config.INPUT_LEN,
        "output_len": config.OUTPUT_LEN,
        "parameters": count_parameters(model),
        "epochs_run": state["epoch"],
        "best_epoch": state["best_epoch"],
        "early_stopped": state["early_stopped"],
        "train_time_sec": state["train_time_sec"],
        "train_only_time_sec": float(sum(r["train_only_sec"] for r in history)),
        "time_per_epoch_sec": state["train_time_sec"] / max(state["epoch"], 1),
        "peak_gpu_mem_mb": max((r["gpu_mem_mb"] for r in history), default=0.0),
        "alpha": alpha_value(model),
        "val": scalar_metrics(val),
        "test": scalar_metrics(test),
        "data": data.summary(),
        "hardware": hardware,
    }
    save_json(metrics, out_dir / "metrics.json")
    np.savez(out_dir / "node_metrics.npz", val_node_mae=val["node_mae"], val_node_rmse=val["node_rmse"],
             test_node_mae=test["node_mae"], test_node_rmse=test["node_rmse"], test_node_count=test["node_count"],
             val_hour_mae=val["hour_mae"], test_hour_mae=test["hour_mae"])
    target_index = kept["start"][:, None] + config.INPUT_LEN + np.arange(config.OUTPUT_LEN)
    np.savez(out_dir / "test_predictions.npz", nodes=data.plot_sensors, pred=kept["pred"], true=kept["true"],
             mask=kept["mask"], start=kept["start"],
             target_time=data.timestamps.to_numpy()[target_index].astype("datetime64[ns]"))
    adjacency = model.current_adjacency()
    if adjacency is not None:
        np.save(out_dir / "adjacency.npy", adjacency.detach().cpu().numpy())
    if args.model == "hybrid":
        np.save(out_dir / "adaptive_component.npy", model.adjacency.adaptive().detach().cpu().numpy())

    log(f"VAL  MAE {val['mae']:.3f} RMSE {val['rmse']:.3f} MAPE {val['mape']:.2f}%")
    log(f"TEST MAE {test['mae']:.3f} RMSE {test['rmse']:.3f} MAPE {test['mape']:.2f}%")
    for horizon, values in test["horizons"].items():
        log(f"TEST {horizon:>6}: MAE {values['mae']:.3f} RMSE {values['rmse']:.3f} MAPE {values['mape']:.2f}%")
    log(f"training time {state['train_time_sec']:.1f}s over {state['epoch']} epochs "
        f"({metrics['time_per_epoch_sec']:.1f}s/epoch), saved results to {out_dir}")


if __name__ == "__main__":
    main()
