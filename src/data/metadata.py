from .io import load_data


def sequential_encoder(df, embedding_keys):
    encoders = {}

    for col in embedding_keys:
        unique_ids = df[col].dropna().unique()
        id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids)}

        encoders[col] = id_to_idx
        df[col] = df[col].map(id_to_idx).fillna(-1).astype(int)

    return df, encoders


def _metadata_columns(embedding_size):
    if isinstance(embedding_size, dict):
        return list(embedding_size.keys())
    return list(embedding_size)


def build_config(df_tax, embedding_size):
    if isinstance(embedding_size, dict):
        return {
            col: (df_tax[col].nunique(), dim)
            for col, dim in embedding_size.items()
        }

    return {
        col: df_tax[col].nunique()
        for col in embedding_size
    }


def load_taxonomy_dataframe(config, embedding_size):
    columns = _metadata_columns(embedding_size)
    df_tax = load_data(config["path"], columns, config.get("cut"))
    df_tax, encoders = sequential_encoder(df_tax, columns)
    config_tax = build_config(df_tax, embedding_size)
    return df_tax, encoders, config_tax
