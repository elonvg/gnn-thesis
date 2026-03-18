import pandas as pd

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