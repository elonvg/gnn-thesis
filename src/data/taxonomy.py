from .io import load_data


def encode_taxonomy(df, embedding_keys):
    encoders = {}

    for col in embedding_keys:
        unique_ids = df[col].dropna().unique()
        id_to_idx = {id_val: idx + 1 for idx, id_val in enumerate(unique_ids)}

        encoders[col] = id_to_idx
        df[col] = df[col].map(id_to_idx).fillna(0).astype(int)

    return df, encoders


def build_taxonomy_config(df_tax, embedding_size):
    return {
        col: (df_tax[col].nunique() + 1, dim)
        for col, dim in embedding_size.items()
    }


def load_taxonomy_dataframe(config, embedding_size):
    df_tax = load_data(config["path"], list(embedding_size.keys()), config.get("cut"))
    df_tax, encoders = encode_taxonomy(df_tax, embedding_size.keys())
    config_tax = build_taxonomy_config(df_tax, embedding_size)
    return df_tax, encoders, config_tax
