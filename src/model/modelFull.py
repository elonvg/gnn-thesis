import torch
import torch.nn as nn

class ToxicityModel(nn.Module):
    def __init__(self, gnn, meta_encoder, gnn_dim, encoder_dim, hidden_dim):
        super().__init__()

        self.gnn = gnn  # plug in a GNN-based encoder here
        self.meta_encoder = meta_encoder
        
        self.predictor = nn.Sequential(
            nn.Linear(gnn_dim + encoder_dim, hidden_dim),  # mol + meta
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        
        mol_embed = self.gnn(data)
        meta_embed = self.meta_encoder(data)

        return self.predictor(torch.cat([mol_embed, meta_embed], dim=1))