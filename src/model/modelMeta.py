import torch
import torch.nn as nn

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