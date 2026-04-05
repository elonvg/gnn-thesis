import torch


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
