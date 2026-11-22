import torch

from src import config

MAPE_MIN_FLOW = 10.0


def masked_mae(pred, true, mask):
    m = mask.float()
    return (torch.abs(pred - true) * m).sum() / m.sum().clamp(min=1.0)


def masked_mse(pred, true, mask):
    m = mask.float()
    return ((pred - true) ** 2 * m).sum() / m.sum().clamp(min=1.0)


class MetricAccumulator:
    def __init__(self, num_nodes, horizons, device):
        shape = (horizons, num_nodes)
        self.abs = torch.zeros(shape, dtype=torch.float64, device=device)
        self.sq = torch.zeros(shape, dtype=torch.float64, device=device)
        self.count = torch.zeros(shape, dtype=torch.float64, device=device)
        self.ape = torch.zeros(shape, dtype=torch.float64, device=device)
        self.ape_count = torch.zeros(shape, dtype=torch.float64, device=device)
        self.hour_abs = torch.zeros((24, horizons), dtype=torch.float64, device=device)
        self.hour_count = torch.zeros((24, horizons), dtype=torch.float64, device=device)

    def update(self, pred, true, mask, hours):
        m = mask.double()
        err = (pred - true).double().abs() * m
        self.abs += err.sum(0)
        self.sq += (err ** 2).sum(0)
        self.count += m.sum(0)
        valid = m * (true.double() > MAPE_MIN_FLOW)
        self.ape += (err / true.double().clamp(min=MAPE_MIN_FLOW) * valid).sum(0)
        self.ape_count += valid.sum(0)
        horizon_idx = torch.arange(pred.shape[1], device=pred.device).expand_as(hours)
        self.hour_abs.index_put_((hours.flatten(), horizon_idx.flatten()), err.sum(2).flatten(), accumulate=True)
        self.hour_count.index_put_((hours.flatten(), horizon_idx.flatten()), m.sum(2).flatten(), accumulate=True)

    def summarize(self, abs_sum, sq_sum, count, ape_sum, ape_count):
        count = max(float(count), 1.0)
        return {
            "mae": float(abs_sum) / count,
            "rmse": (float(sq_sum) / count) ** 0.5,
            "mape": 100.0 * float(ape_sum) / max(float(ape_count), 1.0),
        }

    def results(self):
        out = self.summarize(self.abs.sum(), self.sq.sum(), self.count.sum(), self.ape.sum(), self.ape_count.sum())
        out["horizons"] = {}
        for h in range(self.abs.shape[0]):
            name = f"{config.HORIZON_MINUTES[h]}min" if h < len(config.HORIZON_MINUTES) else f"step{h + 1}"
            out["horizons"][name] = self.summarize(self.abs[h].sum(), self.sq[h].sum(), self.count[h].sum(),
                                                   self.ape[h].sum(), self.ape_count[h].sum())
        node_count = self.count.sum(0).clamp(min=1.0)
        out["node_mae"] = (self.abs.sum(0) / node_count).cpu().numpy()
        out["node_rmse"] = (self.sq.sum(0) / node_count).sqrt().cpu().numpy()
        out["node_count"] = self.count.sum(0).cpu().numpy()
        out["hour_mae"] = (self.hour_abs / self.hour_count.clamp(min=1.0)).cpu().numpy()
        return out


def scalar_metrics(results):
    keep = {k: results[k] for k in ["mae", "rmse", "mape"]}
    keep["horizons"] = results["horizons"]
    return keep
