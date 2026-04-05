import numpy as np

try:
    import deepchem as dc
except ImportError:
    dc = None


def featurize(df, featurizer, apply_filter=False):
    if dc is None:
        raise ImportError("deepchem is required for featurize() but is not installed in this environment.")

    features = featurizer.featurize(df["SMILES"])
    original_size = len(features)

    if apply_filter:
        valid_ids = [i for i, feature in enumerate(features) if isinstance(feature, dc.feat.graph_data.GraphData)]
        features = [features[i] for i in valid_ids]
        df = df[df.index.isin(valid_ids)].reset_index(drop=True)
        print("")
        print(f"Org size: {original_size}, Filtered size: {len(features)}")

    return np.array(features), df
