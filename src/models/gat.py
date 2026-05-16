import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool, global_max_pool

class GATv2(nn.Module):
    def __init__(
            self,
            in_channels=9,
            edge_dim=3,
            hidden_dim=64,
            output_dim=64,
            dropout=0.2
    ):
        super().__init__()

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.lin_in = nn.Linear(in_channels, hidden_dim)

        self.gatconv1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            dropout=dropout, 
            )
        
        self.lin_1 = nn.Linear(hidden_dim, hidden_dim)
        
        self.gatconv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            dropout=dropout,
            )

        self.lin_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Node feature embedding
        x = self.lin_in(x)
        x = F.leaky_relu_(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Gatv2 layers
        x = self.gatconv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin_1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gatconv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x_mean = global_mean_pool(x, batch)
        # x_max = global_max_pool(x, batch)
        # x_add = global_add_pool(x, batch)

        # Concatenate pooled features
        x = torch.cat([x_mean], dim=1)

        x = self.lin_out(x)

        return x
