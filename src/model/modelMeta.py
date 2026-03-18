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

    def forward(self, tax_seq_dict):
        
        # tax_seq_dict: A dict of tensors mapping feature_tax -> sequence of IDs for that feature
        
        embedded_list = []
        for col, emb_layer in self.embeddings.items():
            # Get the IDs for this specific taxonomic rank
            ids = tax_seq_dict[col]
            embedded_list.append(emb_layer(ids))
        
        # Concatenate all embeddings into one vector
        concatenated = torch.cat(embedded_list, dim=-1)

        # Project to the desired output dimension
        return self.projection(concatenated)