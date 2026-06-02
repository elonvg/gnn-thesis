import pandas as pd
import torch


def _normalize_raw_taxid(value):
    if pd.isna(value):
        return 0

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def build_graphs(
    df,
    df_categorical=None,
    categorical_columns=None,
    numerical_columns=None,
):
    graph_objects = []

    df_index = pd.DataFrame(index=df.index)

    for row_idx, graph in enumerate(df["features"]):
        graph.x = graph.x.float()
        graph.y = torch.tensor(df.iloc[row_idx]["log10c"], dtype=torch.float)
        graph.row_id = torch.tensor(row_idx, dtype=torch.long)

        for col in df_index.columns:
            setattr(graph, col, torch.tensor(df_index.iloc[row_idx][col], dtype=torch.long))

        if "taxid" in df.columns:
            graph.taxid_raw = torch.tensor(_normalize_raw_taxid(df.iloc[row_idx]["taxid"]), dtype=torch.long)

        if df_categorical is not None:
            for col in categorical_columns:
                setattr(graph, col, torch.tensor(df_categorical.iloc[row_idx][col], dtype=torch.long))

        for col in numerical_columns:
            setattr(graph, col, torch.tensor(df.iloc[row_idx][col], dtype=torch.float))
        graph_objects.append(graph)

    return graph_objects
