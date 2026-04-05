import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class GCN(nn.Module):
    def __init__(
            self, 
            mol_dim=9, 
            edge_dim=3,
            hidden_dim=64,
            output_dim=64
            ):
        super().__init__()

        self.out_dim = output_dim

        # self.atom_encoder = nn.Embedding(120, 32) # Assuming atomic numbers up to 120 -> embed to 32-dim vectors

        # GCNConv does not accept edge features!!!
        self.conv1 = GCNConv(mol_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.lin1 = nn.Linear(2 * hidden_dim, output_dim)
        

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        # shape: n_atoms x hidden_dim

        # Mean and max pooling
        x_mean = global_mean_pool(x, batch) # Mean pooling captures avg information
        x_max = global_max_pool(x, batch) # Max pooling captures most prominent information

        x = torch.cat([x_mean, x_max], dim=1) # Final molecule embedding of size 2*hidden_dim
        x = self.lin1(x)

        return x