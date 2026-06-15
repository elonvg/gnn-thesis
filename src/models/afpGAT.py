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
        virtual_edge_dim=10,
        hidden_dim=64,
        out_dim=64,
        num_layers=2,
        num_timesteps=2,
        dropout: float=0.2,
    ):
        super().__init__()

        self.lin1 = Linear(in_channels, hidden_dim)

        self.gat = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout,
                            add_self_loops=True,
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

        self.lin3 = Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Inital atom embedding:
        x = F.leaky_relu_(self.lin1(x))

        # First GAT layer
        c = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        c = F.dropout(c, p=self.dropout, training=self.training)
        # Update embedding
        x = self.gru(c, x).relu_()

        # Additional attentive layers
        # Note: using GATConv instead of GATEConv - no edge features
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            c = conv(x, edge_index) # Computer attention + context vector
            c = F.elu(c)
            c = F.dropout(c, p=self.dropout, training=self.training)
            x = gru(c, x).relu() # Updates atom state

        # Molecule embedding:
        row = torch.arange(batch.size(0), device=batch.device) # Atom indices
        edge_index = torch.stack([row, batch], dim=0) # New edge_index for "supernode" molecule

        mol_emb = global_add_pool(x, batch).relu_() # Inital molecule state vector

        # Molecule level refinement - num_timesteps t
        for t in range(self.num_timesteps):
            cmol = F.elu_(self.mol_conv((x, mol_emb), edge_index)) # Attention
            cmol = F.dropout(cmol, p=self.dropout, training=self.training)
            mol_emb = self.mol_gru(cmol, mol_emb).relu_() # Update

        # Output
        mol_emb = F.dropout(mol_emb, p=self.dropout, training=self.training)
        out = self.lin3(out) # Linear layer for final prediction
        return out
