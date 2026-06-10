import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

class GCN(nn.Module):
    def __init__(
            self, 
            in_channels=9, 
            hidden_dim=64,
            out_dim=64,
            dropout=0.3
            ):
        super().__init__()

        self.in_channels=in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = out_dim
        self.dropout = dropout

        # GCNConv does not accept edge features!!!
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.conv5 = GCNConv(hidden_dim, hidden_dim)

        self.lin1 = nn.Linear(in_channels, hidden_dim)
        self.lin2 = nn.Linear(2 * hidden_dim, out_dim)

        

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.lin1(x)
        x = F.leaky_relu_(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = F.leaky_relu_(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index)

        x = F.leaky_relu_(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # shape: n_atoms x hidden_dim

        # Mean and max pooling
        x_mean = global_mean_pool(x, batch) # Mean pooling captures avg information
        x_add = global_add_pool(x, batch) # Max pooling captures most prominent information

        x = torch.cat([x_mean, x_add], dim=1) # Final molecule embedding of size 2*hidden_dim
        x = self.lin2(x)

        return x
