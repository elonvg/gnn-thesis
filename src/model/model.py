import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class ToxicityGNN(nn.Module):

    def __init__(self, atom_feat_dim, metadata_dim, hidden_dim=64):
        super().__init__()

        # Graph branch
        self.conv1 = GCNConv(atom_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Metadata branch — just an embedding + linear for now
        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU()
        )

        # After concat of mean+max pooling + metadata
        # hidden_dim*2 because we concat mean and max pool
        fusion_dim = hidden_dim * 2 + 32
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # single toxicity value
        )

    def forward(self, data, metadata):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN layers
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()

        # Pooling — concat mean and max like DeepChem's GCN did
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_embed = torch.cat([x_mean, x_max], dim=1)

        # Metadata branch
        meta_embed = self.meta_encoder(metadata)

        # Fuse and predict
        out = self.predictor(torch.cat([graph_embed, meta_embed], dim=1))
        return out


class ToxGNN(nn.Module):
    def __init__(self, mol_dim, meta_dim, hidden_dim=64):
        super().__init__()

        self.gnn = GNN(mol_dim, hidden_dim)
        self.meta_encoder = MetaLinear(meta_dim, hidden_dim=64)

        fusion_dim = hidden_dim * 2 + 64 # GNN outputs hidden_dim*2 due to mean+max pooling, and the meta encoder outputs 64
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, data, metadata):
        graph_embed = self.gnn(data)
        meta_embed = self.meta_encoder(metadata)

        out = self.predictor(torch.cat([graph_embed, meta_embed], dim=1))
        return out

class MetaLinear(nn.Module):
    def __init__(self, meta_dim, num_species, hidden_dim=64):
        super().__init__()

        self.species_embedding = nn.Embedding(num_species, 16)

        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, metadata, species_id):
        x = self.meta_encoder(metadata)
        species_embed = self.species_embedding(species_id)

        return torch.cat([x, species_embed], dim=1)

class GNN(nn.Module):
    def __init__(self, mol_dim, hidden_dim=64):
        super().__init__()

        self.atom_encoder = nn.Embedding(120, 32) # Assuming atomic numbers up to 120 -> embed to 32-dim vectors

        self.gnn = nn.Sequential(
            GCNConv(mol_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.gnn(x, edge_index)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_embed = torch.cat([x_mean, x_max], dim=1)

        return graph_embed

