import torch
import torch.nn as nn

class TaxonomyEncoder(nn.Module):
    def __init__(self, config, output_dim=64):
        # config is a dict mapping feature_tax -> (num_unique_values, embedding_dim)
        super().__init__()

        # Create an embedding layer for each taxonomic feature
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_ids, dim) 
            for col, (num_ids, dim) in config.items()
        })
        
        # Calculate the total size of the concatenated vector
        self.raw_dim = sum(dim for _, dim in config.values())

         # Project to fixed output size regardless of which features are selected
        self.projection = nn.Sequential(
            nn.Linear(self.raw_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, data):
        
        embedded_list = []
        for col, emb_layer in self.embeddings.items():
            # Get the IDs for the taxonomic ranks
            ids = getattr(data, col)  # Assuming data has attributes like data.taxid, data.genus, etc.
            embedded_list.append(emb_layer(ids))
        
        # Concatenate all embeddings into one vector
        concatenated = torch.cat(embedded_list, dim=-1)

        # Project to the desired output dimension
        return self.projection(concatenated)
    

class MetaEncoder(nn.Module):
    def __init__ (self, config, meta_dim=1, hidden_dim=16, output_dim=64):
        super().__init__()

        self.encoder_meta = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.encoder_tax = TaxonomyEncoder(config, output_dim=output_dim)

    def forward(self, data):
        duration = data.duration.float().unsqueeze(-1)  # shape: batch_size x 1
        meta_data = self.encoder_meta(duration)
        tax_data = self.encoder_tax(data)

        return torch.cat([meta_data, tax_data], dim=-1)
