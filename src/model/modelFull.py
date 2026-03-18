import torch
import torch.nn as nn

class ToxicityModel(nn.Module):
    def __init__(self, mol_encoder, meta_encoder, hidden_dim):
        super().__init__()
        self.mol_encoder = mol_encoder  # plug in whichever GNN you want
        self.meta_encoder = meta_encoder
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # mol + meta
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, data, metadata):
        mol_embed = self.mol_encoder(data)
        meta_embed = self.meta_encoder(metadata)
        return self.predictor(torch.cat([mol_embed, meta_embed], dim=1))