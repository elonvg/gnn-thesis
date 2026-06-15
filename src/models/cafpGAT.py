from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter, LayerNorm

from torch_geometric.nn import (
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)


class cAFPGAT(torch.nn.Module):

    def __init__(
        self,
        in_channels: int = 9,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 3,
        num_timesteps: int = 2,
        num_heads: int = 4,          # [NEW] multi-head attention
        dropout: float = 0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.num_heads = num_heads
        self.dropout = dropout

        # ── Atom embedding ──────────────────────────────────────────────────
        self.lin1 = Linear(in_channels, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)                  # [1] stabilise

        # ── First atom GAT+GRU ──────────────────────────────────────────────
        # concat=False → output stays at hidden_dim (heads average, not concat)
        self.gat = GATv2Conv(
            hidden_dim, hidden_dim,
            heads=num_heads, concat=False,                  # [2] multi-head
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
            negative_slope=0.01,
        )
        self.gru = GRUCell(hidden_dim, hidden_dim)

        # ── Additional atom GAT+GRU layers ─────────────────────────────────
        self.atom_gats = nn.ModuleList()
        self.atom_grus = nn.ModuleList()
        for _ in range(num_layers - 1):
            gat = GATv2Conv(
                hidden_dim, hidden_dim,
                heads=num_heads, concat=False,              # [2] multi-head
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=False,
                negative_slope=0.01,
            )
            gru = GRUCell(hidden_dim, hidden_dim)
            self.atom_gats.append(gat)
            self.atom_grus.append(gru)

        # ── Triple-pool projection ───────────────────────────────────────────
        # Concat(add, mean, max) gives 3*hidden_dim → project back to hidden_dim
        self.pool_proj = nn.Sequential(               # [4] multi-pool readout
            LayerNorm(3 * hidden_dim),
            Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # ── Molecule-level refinement ────────────────────────────────────────
        # No edge_dim here: the supernode graph has no bond features, same as
        # original. We pass aggregated atom features as a separate channel via
        # the mol_edge_proj below instead of forcing edge_attr into mol_gat.
        self.mol_gat = GATv2Conv(
            hidden_dim, hidden_dim,
            heads=num_heads, concat=False,                  # [2] multi-head
            dropout=dropout,
            add_self_loops=False,
            negative_slope=0.01,
        )
        self.mol_gru = GRUCell(hidden_dim, hidden_dim)

        # ── MLP prediction head ──────────────────────────────────────────────
        self.head = nn.Sequential(                          # [6] deeper head
            LayerNorm(hidden_dim),
            Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, out_dim),
        )

    # ────────────────────────────────────────────────────────────────────────
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.norm1.reset_parameters()
        self.gat.reset_parameters()
        self.gru.reset_parameters()
        for gat, gru in zip(self.atom_gats, self.atom_grus):
            gat.reset_parameters()
            gru.reset_parameters()
        for m in self.pool_proj:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        self.mol_gat.reset_parameters()
        self.mol_gru.reset_parameters()
        for m in self.head:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    # ────────────────────────────────────────────────────────────────────────
    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # [1] Initial atom embedding with LayerNorm
        x = self.norm1(F.leaky_relu(self.lin1(x), negative_slope=0.01))

        # ── First atom GAT layer ────────────────────────────────────────────
        x_prev = x
        c = F.elu(self.gat(x, edge_index, edge_attr))
        c = F.dropout(c, p=self.dropout, training=self.training)
        x = F.relu(self.gru(c, x) + x_prev)              # [3] residual

        # ── Additional atom layers ──────────────────────────────────────────
        for gat, gru in zip(self.atom_gats, self.atom_grus):
            x_prev = x
            c = F.elu(gat(x, edge_index, edge_attr))
            c = F.dropout(c, p=self.dropout, training=self.training)
            x = F.relu(gru(c, x) + x_prev)               # [3] residual

        # [4] Triple pooling → project to hidden_dim
        mol_emb = self.pool_proj(
            torch.cat([
                global_add_pool(x, batch),
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
            ], dim=-1)
        )

        # ── Molecule-level refinement ───────────────────────────────────────
        # Build the atom→supernode edge_index (same as original)
        row = torch.arange(batch.size(0), device=batch.device)
        mol_edge_index = torch.stack([row, batch], dim=0)

        for _ in range(self.num_timesteps):
            cmol = F.elu(self.mol_gat((x, mol_emb), mol_edge_index))
            cmol = F.dropout(cmol, p=self.dropout, training=self.training)
            mol_emb = F.relu(self.mol_gru(cmol, mol_emb))

        # [6] MLP head
        mol_emb = F.dropout(mol_emb, p=self.dropout, training=self.training)
        return self.head(mol_emb)