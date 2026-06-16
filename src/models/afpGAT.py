from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter

from torch_geometric.nn import (
    GATv2Conv,
    global_add_pool, 
    global_mean_pool,
    global_max_pool
)



class AFPGAT(torch.nn.Module):

    def __init__(
        self,
        in_channels=9,
        edge_dim=3,
        hidden_dim=64,
        out_dim=64,
        num_layers=3,
        num_timesteps=2,
        dropout: float=0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        # self.lin1 = Linear(in_channels, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout,
                            add_self_loops=False,
                            negative_slope=0.01) # negative_slope=0.01 for leakyReLU, to suppress negative values

        self.gru = GRUCell(hidden_dim, hidden_dim)

        # GAT Layers
        self.atom_gats = nn.ModuleList() 
        self.atom_grus = nn.ModuleList()
        for _ in range(num_layers - 1):
            gat = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout,
                             add_self_loops=False, negative_slope=0.01)
            
            gru = GRUCell(hidden_dim, hidden_dim)

            self.atom_gats.append(gat)
            self.atom_grus.append(gru)  

        self.mol_gat = GATv2Conv(hidden_dim, hidden_dim, dropout=dropout,
                                  add_self_loops=False, negative_slope=0.01)
        
        self.mol_gru = GRUCell(hidden_dim, hidden_dim)

        self.lin2 = Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.norm1.reset_parameters()
        self.gat.reset_parameters()
        self.gru.reset_parameters()
        for gat, gru in zip(self.atom_gats, self.atom_grus):
            gat.reset_parameters()
            gru.reset_parameters()
        self.mol_gat.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Inital atom embedding:
        x = F.leaky_relu_(self.lin1(x))
        x = self.norm1(x)

        # First GAT layer
        x_prev = x
        c = F.elu_(self.gat(x, edge_index, edge_attr))
        c = F.dropout(c, p=self.dropout, training=self.training)
        # Update embedding
        x = F.relu(self.gru(c, x) + x_prev) # Residual connection

        # Additional attentive layers
        for gat, gru in zip(self.atom_gats, self.atom_grus):
            x = x_prev
            c = gat(x, edge_index, edge_attr) # Computer attention + context vector
            c = F.elu(c)
            c = F.dropout(c, p=self.dropout, training=self.training)
            x = F.relu(self.gru(c, x) + x_prev)

        # Molecule embedding:
        row = torch.arange(batch.size(0), device=batch.device) # Atom indices
        edge_index = torch.stack([row, batch], dim=0) # New edge_index for "supernode" molecule

        mol_emb = global_add_pool(x, batch).relu_() # Inital molecule state vector

        # Molecule level refinement - num_timesteps t
        for t in range(self.num_timesteps):
            cmol = F.elu_(self.mol_gat((x, mol_emb), edge_index)) # Attention
            cmol = F.dropout(cmol, p=self.dropout, training=self.training)
            mol_emb = self.mol_gru(cmol, mol_emb).relu_() # Update

        # Output
        mol_emb = F.dropout(mol_emb, p=self.dropout, training=self.training)

        out = self.lin2(mol_emb) # Linear layer for final prediction
        return out
