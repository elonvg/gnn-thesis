import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class GIN(nn.Module):
    def __init__(
            self,
            mol_dim=9,
            edge_dim=3,
            num_layers=3,
            hidden_dim=64,
            output_dim=64
            ):
        
        super().__init__()

        self.init = GINEConv(
            nn.Sequential(
                nn.Linear(mol_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ),
            edge_dim=edge_dim
        )

        self.mp_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.mp_layers.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim)
                    ),
                    edge_dim=edge_dim
                )
            )

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch

        x = self.init(x, edge_index, edge_attr)
        for mp in self.mp_layers:
            x = mp(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch) # shape: n_molecules x hidden_dim

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)

        return x