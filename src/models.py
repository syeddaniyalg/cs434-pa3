import torch
import torch.nn as nn
import torch.nn.functional as F

from src.adaptive import AdaptiveAdjacency, FixedAdjacency, HybridAdjacency
from src.graph import choose_normalization, normalize_adjacency


class TemporalEncoder(nn.Module):
    def __init__(self, input_len, hidden):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2)
        self.proj = nn.Linear(hidden * input_len, hidden)

    def forward(self, x):
        b, t, n = x.shape
        x = x.permute(0, 2, 1).reshape(b * n, 1, t)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        z = F.relu(self.proj(x.reshape(b * n, -1)))
        return z.reshape(b, n, -1)


class GraphConv(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=False)

    def forward(self, z, adj=None):
        if adj is not None:
            z = torch.einsum("nm,bmd->bnd", adj, z)
        return F.relu(self.linear(z))


class ForecastHead(nn.Module):
    def __init__(self, hidden, output_len):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output_len))

    def forward(self, h):
        return self.net(h).permute(0, 2, 1)


class STGNN(nn.Module):
    def __init__(self, input_len, output_len, hidden, graph_layers=1, dropout=0.0, adjacency=None):
        super().__init__()
        self.encoder = TemporalEncoder(input_len, hidden)
        self.adjacency = adjacency
        self.graph_layers = nn.ModuleList([GraphConv(hidden) for _ in range(graph_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = ForecastHead(hidden, output_len)

    def current_adjacency(self):
        return None if self.adjacency is None else self.adjacency()

    def forward(self, x):
        h = self.encoder(x)
        adj = self.current_adjacency()
        for layer in self.graph_layers:
            h = h + layer(h, adj)
        return self.head(self.dropout(h))


def build_model(model_type, raw_adj, num_nodes, input_len, output_len, cfg, source_mask=None):
    adjacency = None
    norm = None
    if model_type == "fixed":
        norm = choose_normalization(raw_adj)
        adjacency = FixedAdjacency(normalize_adjacency(raw_adj, norm))
    elif model_type == "adaptive":
        adjacency = AdaptiveAdjacency(num_nodes, cfg["embed_dim"], cfg["topk"], source_mask)
    elif model_type == "hybrid":
        norm = "row"
        adjacency = HybridAdjacency(normalize_adjacency(raw_adj, norm), num_nodes, cfg["embed_dim"],
                                    cfg["topk"], cfg["alpha"], source_mask)
    elif model_type != "temporal":
        raise ValueError(f"unknown model type {model_type}")
    model = STGNN(input_len, output_len, cfg["hidden"], cfg["graph_layers"], cfg["dropout"], adjacency)
    return model, norm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
