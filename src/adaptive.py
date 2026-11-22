import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedAdjacency(nn.Module):
    def __init__(self, adj):
        super().__init__()
        self.register_buffer("adj", torch.as_tensor(adj, dtype=torch.float32))

    def forward(self):
        return self.adj


class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes, embed_dim, topk=0, source_mask=None):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.e2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.topk = topk
        if source_mask is None:
            source_mask = torch.ones(num_nodes, dtype=torch.bool)
        self.register_buffer("source_mask", torch.as_tensor(source_mask, dtype=torch.bool))

    def scores(self):
        return F.relu(self.e1 @ self.e2.T)

    def forward(self):
        scores = self.scores().masked_fill(~self.source_mask[None, :], float("-inf"))
        if self.topk and self.topk < scores.shape[1]:
            values, index = scores.topk(self.topk, dim=1)
            scores = torch.full_like(scores, float("-inf")).scatter(1, index, values)
        return torch.softmax(scores, dim=1)


class HybridAdjacency(nn.Module):
    def __init__(self, fixed_adj, num_nodes, embed_dim, topk=0, alpha="learn", source_mask=None):
        super().__init__()
        self.fixed = FixedAdjacency(fixed_adj)
        self.adaptive = AdaptiveAdjacency(num_nodes, embed_dim, topk, source_mask)
        self.learn_alpha = alpha == "learn"
        if self.learn_alpha:
            self.alpha_logit = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("alpha_value", torch.tensor([float(alpha)]))

    def alpha(self):
        return torch.sigmoid(self.alpha_logit) if self.learn_alpha else self.alpha_value

    def forward(self):
        a = self.alpha()
        return a * self.fixed() + (1 - a) * self.adaptive()
