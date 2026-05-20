import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter

from torch_geometric.nn import (
    GATv2Conv,
    PNAConv, 
    global_add_pool, 
    global_mean_pool,
    global_max_pool
)
from src import data

DEFAULT_AGGREGATORS = ("mean", "min", "max", "std")
DEFAULT_SCALERS = ("identity", "amplification", "attenuation")

class ComboPNALayer(nn.Module):

    def __init__(
            self,
            in_channels=9,
            edge_dim=3,
            hidden_dim=64,
            aggregators=DEFAULT_AGGREGATORS,
            scalers=DEFAULT_SCALERS,
            deg=None,
            out_dim=64,
            num_layers=2,
            num_timesteps=2,
            dropout: float=0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.aggregators = aggregators
        self.scalers = scalers
        self.deg = deg
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_dim)

        self.gat = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout,
                            add_self_loops=False, # add_self_loops=False since GRU will handle self-loops
                            negative_slope=0.01) # negative_slope=0.01 for leakyReLU, to suppress negative values
        self.gru = GRUCell(hidden_dim, hidden_dim)

        # PNA Layers
        self.atom_pnas = nn.ModuleList()
        self.atom_grus = nn.ModuleList()
        for _ in range(num_layers - 1):
            pna = PNAConv(in_channels=hidden_dim,
                          out_channels=hidden_dim, 
                          edge_dim=edge_dim, 
                          aggregators=self.aggregators,
                          scalers=self.scalers,
                          deg=self.deg,
                        )
            
            gru = GRUCell(hidden_dim, hidden_dim)

            self.atom_pnas.append(pna)
            self.atom_grus.append(gru)  

        self.mol_gat = GATv2Conv(hidden_dim, hidden_dim, dropout=dropout,
                                  add_self_loops=False, negative_slope=0.01)
        # self.mol_gat.explain = False # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_dim, hidden_dim)

        self.lin2 = Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gat.reset_parameters()
        self.gru.reset_parameters()
        for pna, gru in zip(self.atom_pnas, self.atom_grus):
            pna.reset_parameters()
            gru.reset_parameters()
        self.mol_gat.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Initial atom embedding
        x = self.lin1(x)
        x = F.leaky_relu(x)

        # Attention layers
        c = self.gat(x, edge_index, edge_attr) # Context vector
        c = F.elu(c)
        c = F.dropout(c, p=self.dropout, training=self.training)

        # Recursive update
        x = self.gru(c, x)
        x = F.relu(x)

        # PNA Layers
        for pna, gru in zip(self.atom_pnas, self.atom_grus):
            pc= pna(x, edge_index, edge_attr)
            pc = F.elu(pc)
            pc = F.dropout(pc, p=self.dropout, training=self.training)
            x = gru(pc, x)
            x = F.relu(x)

        # Molecule embedding
        row = torch.arange(batch.size(0), device=batch.device) # Atom indices
        edge_index = torch.stack([row, batch], dim=0) # New edge_index for "supernode" molecule
        
        mol = global_add_pool(x, batch).relu_() # Initial molecule state vector

        # Repeat for num_timesteps
        for t in range(self.num_timesteps):
            c = self.mol_gat((x, mol), edge_index) # Attention
            c = F.elu(c)
            c = F.dropout(c, p=self.dropout, training=self.training)
            mol = self.mol_gru(c, mol)
            mol = F.relu(mol)

        out = F.dropout(mol, p=self.dropout, training=self.training)
        out = self.lin2(out)

        return out
