import pandas as pd
import torch


def _normalize_raw_taxid(value):
    if pd.isna(value):
        return 0

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def build_graph_features(
    df,
    df_tax,
    tax_embedding,
    df_categorical=None,
    categorical_columns=None,
    numerical_columns=None,
):
    features = []

    if df_categorical is not None and categorical_columns is None:
        categorical_columns = list(df_categorical.columns)
    if numerical_columns is None:
        numerical_columns = ["duration"]

    for row_idx, graph in enumerate(df["features"]):
        graph.x = graph.x.float()
        graph.y = torch.tensor(df.iloc[row_idx]["log10c"], dtype=torch.float)

        for col in df_tax.columns:
            setattr(graph, col, torch.tensor(df_tax.iloc[row_idx][col], dtype=torch.long))

        if "taxid" in df.columns:
            graph.taxid_raw = torch.tensor(_normalize_raw_taxid(df.iloc[row_idx]["taxid"]), dtype=torch.long)

        if df_categorical is not None:
            for col in categorical_columns:
                setattr(graph, col, torch.tensor(df_categorical.iloc[row_idx][col], dtype=torch.long))

        for col in numerical_columns:
            setattr(graph, col, torch.tensor(df.iloc[row_idx][col], dtype=torch.float))
        features.append(graph)

    return features
