import pandas as pd
import torch

from .dataMol import load_data

def encode_taxonomy(df, embedding_keys):
    encoders = {}  # save mappings for inference later
    
    for col in embedding_keys:
        # Get unique IDs and create a mapping to sequential integers
        # 0 is reserved for unknown
        unique_ids = df[col].dropna().unique()

        id_to_idx = {id_val: idx + 1 for idx, id_val in enumerate(unique_ids)}
        
        encoders[col] = id_to_idx
        df[col] = df[col].map(id_to_idx).fillna(0).astype(int)
    
    return df, encoders

def load_taxonomy_dataframe(config, embedding_size):
    df_tax = load_data(config["path"], embedding_size.keys(), config["cut"])
    df_tax, encoders = encode_taxonomy(df_tax, embedding_size.keys())
    config_tax = {
        col: (df_tax[col].nunique() + 1, dim)
        for col, dim in embedding_size.items()
    }
    return df_tax, encoders, config_tax

def build_graph_features(df, df_tax, embedding_size):
    features = []

    for row_idx, graph in enumerate(df["features"]):
        graph.x = graph.x.float()
        graph.y = torch.tensor(df.iloc[row_idx]["log10c"], dtype=torch.float)

        for col in embedding_size.keys():
            setattr(graph, col, torch.tensor(df_tax.iloc[row_idx][col], dtype=torch.long))

        graph.duration = torch.tensor(df.iloc[row_idx]["duration"], dtype=torch.float)
        features.append(graph)

    return features
