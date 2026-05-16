from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    PNAConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import degree


PNA_1_5M_CONFIG = {
    "hidden_dim": 256,
    "output_dim": 128,
    "num_layers": 4,
    "towers": 4,
    "pre_layers": 1,
    "post_layers": 1,
    "dropout": 0.25,
}

DEFAULT_AGGREGATORS = ("mean", "min", "max", "std")
DEFAULT_SCALERS = ("identity", "amplification", "attenuation")
DEFAULT_MOLECULAR_DEGREE_HISTOGRAM = torch.tensor(
    [1, 8, 24, 16, 4, 1],
    dtype=torch.long,
)


def compute_pna_degree_histogram(dataset_or_loader: Iterable) -> torch.Tensor:
    """Compute the in-degree histogram expected by PyG's PNAConv."""
    deg_histogram = torch.zeros(1, dtype=torch.long)

    for data in dataset_or_loader:
        deg = degree(
            data.edge_index[1],
            num_nodes=data.num_nodes,
            dtype=torch.long,
        )
        deg_bincount = torch.bincount(deg.cpu(), minlength=deg_histogram.numel())

        if deg_bincount.numel() > deg_histogram.numel():
            deg_bincount[: deg_histogram.numel()] += deg_histogram
            deg_histogram = deg_bincount
        else:
            deg_histogram += deg_bincount

    return deg_histogram


class PNA(nn.Module):
    """Principal Neighbourhood Aggregation GNN for molecular graph embeddings.

    With the default 9 atom features and 3 edge features, the
    ``PNA_1_5M_CONFIG`` settings give roughly 1.55M trainable GNN parameters.
    Pass a training-set degree histogram through ``deg`` whenever possible.
    """

    def __init__(
        self,
        mol_dim=9,
        edge_dim=3,
        hidden_dim=256,
        output_dim=128,
        num_layers=4,
        towers=4,
        pre_layers=1,
        post_layers=1,
        dropout=0.25,
        deg=None,
        aggregators=None,
        scalers=None,
        readout_hidden_dim=None,
        pooling=("mean", "max", "add"),
        use_residual=True,
    ):
        super().__init__()

        if hidden_dim % towers != 0:
            raise ValueError("hidden_dim must be divisible by towers for PNAConv.")

        self.mol_dim = mol_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.out_dim = output_dim
        self.num_layers = num_layers
        self.towers = towers
        self.pre_layers = pre_layers
        self.post_layers = post_layers
        self.dropout = dropout
        self.pooling = tuple(pooling)
        self.use_residual = use_residual

        self.aggregators = list(aggregators or DEFAULT_AGGREGATORS)
        self.scalers = list(scalers or DEFAULT_SCALERS)
        deg_tensor = torch.as_tensor(
            deg if deg is not None else DEFAULT_MOLECULAR_DEGREE_HISTOGRAM,
            dtype=torch.long,
        )
        self.register_buffer("deg", deg_tensor)

        self.lin_in = nn.Linear(mol_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=self.aggregators,
                    scalers=self.scalers,
                    deg=self.deg,
                    edge_dim=edge_dim,
                    towers=towers,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    divide_input=True,
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        readout_hidden_dim = readout_hidden_dim or hidden_dim
        readout_dim = len(self.pooling) * hidden_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, readout_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden_dim, output_dim),
        )

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        for module in self.readout:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _pool(self, x, batch):
        pooled = []
        for pool in self.pooling:
            if pool == "mean":
                pooled.append(global_mean_pool(x, batch))
            elif pool == "max":
                pooled.append(global_max_pool(x, batch))
            elif pool in {"add", "sum"}:
                pooled.append(global_add_pool(x, batch))
            else:
                raise ValueError(f"Unknown pooling mode: {pool!r}")
        return torch.cat(pooled, dim=1)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        edge_attr = edge_attr.float() if edge_attr is not None else None
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + residual

        x = self._pool(x, batch)
        return self.readout(x)


PNAGNN = PNA


__all__ = [
    "PNA",
    "PNAGNN",
    "PNA_1_5M_CONFIG",
    "compute_pna_degree_histogram",
]
