from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter

from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax


class GATEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # edge_updater_type: (x: Tensor, edge_attr: Tensor)
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        # propagate_type: (x: Tensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out + self.bias
        return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                    index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class AFPFlex(torch.nn.Module):

    def __init__(
        self,
        in_channels=9,
        edge_dim=3,
        hidden_channels=64,
        out_channels=64,
        num_layers=3,
        num_timesteps=2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.out_dim = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.linlone = nn.Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

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
        self.linlone.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        num_graphs = int(batch.max()) + 1
        has_edges = torch.zeros(num_graphs, dtype=torch.bool, device=batch.device)
        # Count number of edges for each graph
        if edge_index.numel():
            has_edges[batch[edge_index[0]]] = True

        # Inital atom embedding:
        x0 = F.leaky_relu_(self.lin1(x))

        # First attentive layer - using edge features
        h = F.elu_(self.gate_conv(x0, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        # Add parts of attention info to atom embedding
        xg = self.gru(h, x0).relu_() # h is C in paper

        # Additional attentive layers
        # Note: using GATConv instead of GATEConv - no edge features
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(xg, edge_index) # Computer attention + context vector
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            xg = gru(h, xg).relu() # Updates atom state
    
        # Molecule embedding:
        row = torch.arange(batch.size(0), device=batch.device) # Atom indices
        edge_index = torch.stack([row, batch], dim=0) # New edge_index for "supernode" molecule

        out = global_add_pool(xg, batch).relu_() # Inital molecule state vector

        # Molecule level refinement - num_timesteps t
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((xg, out), edge_index)) # Attention
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_() # Update

        gnn_out = out
        
        # Output for edgess graphs
        xl = self.linlone(x0)
        atom_out = global_add_pool(xl, batch) # Sum all state vectors that belong to same graph

        # Predictor:
        out = torch.where(has_edges.unsqueeze(-1), gnn_out, atom_out) # Filter for graphs with/without edges
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        pred = self.lin2(out) # Linear layer for final prediction
        return pred
