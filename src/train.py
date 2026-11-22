import csv
import math
import random
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from src import config
from src.metrics import MetricAccumulator, masked_mae


@torch.no_grad()
def evaluate(model, data, split, batch_size, keep_nodes=None, desc=None):
    model.eval()
    acc = MetricAccumulator(data.num_nodes, config.OUTPUT_LEN, data.device)
    kept = {"pred": [], "true": [], "mask": [], "start": []}
    bar = tqdm(data.batches(split, batch_size), total=data.num_batches(split, batch_size),
               desc=desc or f"eval {split}", leave=False, unit="batch")
    for x, y, m, s in bar:
        pred = data.scaler.inverse(model(x))
        acc.update(pred, y, m, data.target_hours(s))
        if keep_nodes is not None:
            kept["pred"].append(pred[:, :, keep_nodes].cpu())
            kept["true"].append(y[:, :, keep_nodes].cpu())
            kept["mask"].append(m[:, :, keep_nodes].cpu())
            kept["start"].append(s.cpu())
    results = acc.results()
    if keep_nodes is None:
        return results, None
    return results, {k: torch.cat(v).numpy() for k, v in kept.items()}


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, state):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "state": state, "rng": rng_state()}, tmp)
    shutil.move(str(tmp), str(path))


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def alpha_value(model):
    adjacency = getattr(model, "adjacency", None)
    if adjacency is not None and hasattr(adjacency, "alpha"):
        return float(adjacency.alpha().item())
    return None


def write_history(history, path):
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def new_state():
    return {"epoch": 0, "best_val_mae": math.inf, "best_epoch": 0, "bad_epochs": 0,
            "train_time_sec": 0.0, "history": [], "early_stopped": False}


def make_optimizer(model, cfg):
    graph_params, other_params = [], []
    for name, param in model.named_parameters():
        (graph_params if name.startswith("adjacency.") else other_params).append(param)
    groups = [{"params": other_params, "weight_decay": cfg["weight_decay"]}]
    if graph_params:
        groups.append({"params": graph_params, "weight_decay": 0.0})
    return torch.optim.Adam(groups, lr=cfg["lr"])


def train_model(model, data, cfg, run_dir, log):
    optimizer = make_optimizer(model, cfg)
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    state = new_state()
    if last_path.exists():
        ckpt = load_checkpoint(last_path, model, optimizer)
        state = ckpt["state"]
        set_rng_state(ckpt["rng"])
        log(f"resumed from {last_path} at epoch {state['epoch']} (best val MAE {state['best_val_mae']:.3f})")
    if state["early_stopped"] or state["epoch"] >= cfg["epochs"]:
        log("training already finished, skipping to evaluation")
        return state

    steps = data.num_batches("train", cfg["batch_size"])
    epoch_bar = tqdm(range(state["epoch"] + 1, cfg["epochs"] + 1), initial=state["epoch"], total=cfg["epochs"],
                     desc=f"{run_dir.name} epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        start = time.time()
        loss_sum = 0.0
        batch_bar = tqdm(data.batches("train", cfg["batch_size"], shuffle=True, seed=config.SEED + epoch),
                         total=steps, desc=f"epoch {epoch}", leave=False, unit="batch")
        for i, (x, y, m, _) in enumerate(batch_bar, start=1):
            pred = data.scaler.inverse(model(x))
            loss = masked_mae(pred, y, m)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            optimizer.step()
            loss_sum += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{loss_sum / i:.3f}", grad=f"{grad_norm:.2f}")
        train_seconds = time.time() - start
        val, _ = evaluate(model, data, "val", cfg["eval_batch_size"], desc=f"val {epoch}")
        epoch_seconds = time.time() - start
        state["train_time_sec"] += epoch_seconds
        state["epoch"] = epoch

        improved = val["mae"] < state["best_val_mae"]
        if improved:
            state["best_val_mae"] = val["mae"]
            state["best_epoch"] = epoch
            state["bad_epochs"] = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val["mae"]}, best_path)
        else:
            state["bad_epochs"] += 1

        record = {
            "epoch": epoch,
            "train_mae": loss_sum / steps,
            "val_mae": val["mae"],
            "val_rmse": val["rmse"],
            "val_mape": val["mape"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_sec": epoch_seconds,
            "train_only_sec": train_seconds,
            "cumulative_sec": state["train_time_sec"],
            "alpha": alpha_value(model),
            "gpu_mem_mb": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
            "improved": improved,
        }
        state["history"].append(record)
        epoch_bar.set_postfix(train=f"{record['train_mae']:.3f}", val_mae=f"{val['mae']:.3f}",
                              val_rmse=f"{val['rmse']:.3f}", best=f"{state['best_val_mae']:.3f}",
                              bad=state["bad_epochs"])
        alpha_text = f" alpha {record['alpha']:.4f}" if record["alpha"] is not None else ""
        log(f"epoch {epoch:3d} | train MAE {record['train_mae']:.3f} | val MAE {val['mae']:.3f} "
            f"RMSE {val['rmse']:.3f} MAPE {val['mape']:.2f}% | {epoch_seconds:.1f}s{alpha_text}"
            f"{' | best' if improved else ''}")

        stop = state["bad_epochs"] >= cfg["patience"]
        state["early_stopped"] = stop
        if improved or stop or epoch % cfg["ckpt_every"] == 0 or epoch == cfg["epochs"]:
            save_checkpoint(last_path, model, optimizer, state)
        write_history(state["history"], run_dir / "history.csv")
        if stop:
            log(f"early stopping at epoch {epoch}, best epoch {state['best_epoch']}")
            break
    return state
