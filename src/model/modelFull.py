import torch
import torch.nn as nn

class ToxicityModel(nn.Module):
    def __init__(self, gnn, encoder_tax, hidden_dim):
        super().__init__()

        self.gnn = gnn  # plug in a GNN-based encoder here
        self.encoder_tax = encoder_tax
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # mol + meta
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, data):
        
        mol_embed = self.gnn(data)
        meta_embed = self.encoder_tax(data)

        return self.predictor(torch.cat([mol_embed, meta_embed], dim=1))