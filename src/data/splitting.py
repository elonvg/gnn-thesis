from collections import defaultdict, deque
from functools import lru_cache
import hashlib
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import rdkit
import torch
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from skfp.model_selection import butina_train_test_split
except ImportError:
    butina_train_test_split = None


def generate_scaffold(smiles, include_chirality=False, radius=2):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)

    if scaffold == "":
        info = {}
        AllChem.GetMorganFingerprint(mol, radius=radius, bitInfo=info)
        scaffold = hashlib.md5(str(sorted(info.keys())).encode()).hexdigest()

    return scaffold


def scaffold_split(features, frac_train=0.8, frac_test=0.2, frac_valid=0.0, seed=None):
    assert abs(frac_train + frac_test + frac_valid - 1.0) < 1e-6

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    n = len(features)
    all_scaffolds = {}
    for i in range(n):
        scaffold = generate_scaffold(smiles=features[i].smiles, include_chirality=False)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    scaffold_sets = sorted(all_scaffolds.values(), key=lambda x: len(x), reverse=True)

    train_indices, test_indices, val_indices = [], [], []
    train_cutoff = frac_train * n
    val_cutoff = (frac_train + frac_valid) * n

    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= train_cutoff:
            train_indices.extend(scaffold_set)
        elif len(train_indices) + len(val_indices) + len(scaffold_set) <= val_cutoff:
            val_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    train_dataset = [features[i] for i in train_indices]
    test_dataset = [features[i] for i in test_indices]
    val_dataset = [features[i] for i in val_indices]

    return train_dataset, test_dataset, val_dataset


def scaffold_split_ac(features, frac_train=0.8, frac_test=0.2, frac_valid=0.0, seed=None):
    assert abs(frac_train + frac_test + frac_valid - 1.0) < 1e-6

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    n = len(features)
    all_scaffolds = {}
    acyclic_indices = []

    for i in range(n):
        scaffold = generate_scaffold(smiles=features[i].smiles, include_chirality=False)
        if scaffold == "":
            acyclic_indices.append(i)
            continue
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    scaffold_sets = sorted(all_scaffolds.values(), key=lambda x: len(x), reverse=True)

    train_indices, test_indices, val_indices = [], [], []
    train_cutoff = frac_train * n
    val_cutoff = (frac_train + frac_valid) * n

    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= train_cutoff:
            train_indices.extend(scaffold_set)
        elif len(train_indices) + len(val_indices) + len(scaffold_set) <= val_cutoff:
            val_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    random.shuffle(acyclic_indices)
    n_acyclic = len(acyclic_indices)
    n_train = round(frac_train * n_acyclic)
    n_val = round(frac_valid * n_acyclic)

    train_indices.extend(acyclic_indices[:n_train])
    val_indices.extend(acyclic_indices[n_train:n_train + n_val])
    test_indices.extend(acyclic_indices[n_train + n_val:])

    train_dataset = [deepcopy(features[i]) for i in train_indices]
    test_dataset = [deepcopy(features[i]) for i in test_indices]
    val_dataset = [deepcopy(features[i]) for i in val_indices]

    return train_dataset, test_dataset, val_dataset


def _build_smiles_index_lookup(smiles_list):
    indices_by_smile = defaultdict(deque)

    for idx, smile in enumerate(smiles_list):
        indices_by_smile[smile].append(idx)

    return indices_by_smile


def _take_split_indices(split_smiles, indices_by_smile):
    split_indices = []

    for smile in split_smiles:
        if not indices_by_smile[smile]:
            raise ValueError(
                f"Could not map split SMILES {smile!r} back to a unique row index. "
                "The split output does not match the original feature list."
            )
        split_indices.append(indices_by_smile[smile].popleft())

    return split_indices


def _subset_features(features, indices):
    return [deepcopy(features[i]) for i in indices]


def _butina_train_test_indices(features, train_size, test_size):
    smiles_list = [g.smiles for g in features]
    y_list = [g.y for g in features]

    smiles_train, smiles_test, _, _ = butina_train_test_split(
        smiles_list,
        y_list,
        train_size=train_size,
        test_size=test_size,
    )

    indices_by_smile = _build_smiles_index_lookup(smiles_list)
    train_indices = _take_split_indices(smiles_train, indices_by_smile)
    test_indices = _take_split_indices(smiles_test, indices_by_smile)

    return train_indices, test_indices


@lru_cache(maxsize=None)
def load_precomputed_butina_clusters(cluster_csv_path, cluster_column="Cluster_at_cutoff_0.3"):
    try:
        cluster_df = pd.read_csv(
            cluster_csv_path,
            usecols=["SMILES", cluster_column],
            compression="infer",
            low_memory=False,
        )
    except ValueError as exc:
        available_columns = pd.read_csv(
            cluster_csv_path,
            nrows=0,
            compression="infer",
        ).columns.tolist()
        raise ValueError(
            f"Column {cluster_column!r} was not found in {cluster_csv_path!r}. "
            f"Available columns: {available_columns}"
        ) from exc

    return dict(zip(cluster_df["SMILES"], cluster_df[cluster_column]))


def _precomputed_cluster_key(smiles, smiles_to_cluster):
    cluster_id = smiles_to_cluster.get(smiles)

    if pd.isna(cluster_id):
        return f"__missing__::{smiles}"

    return f"cluster::{cluster_id}"


def _precomputed_butina_train_test_indices(
    features,
    train_size,
    test_size,
    cluster_csv_path,
    cluster_column="Cluster_at_cutoff_0.3",
):
    smiles_to_cluster = load_precomputed_butina_clusters(cluster_csv_path, cluster_column)
    indices_by_cluster = defaultdict(list)

    for idx, feature in enumerate(features):
        cluster_key = _precomputed_cluster_key(feature.smiles, smiles_to_cluster)
        indices_by_cluster[cluster_key].append(idx)

    cluster_sets = sorted(indices_by_cluster.values(), key=lambda indices: (-len(indices), indices[0]))

    train_indices, test_indices = [], []
    train_cutoff = train_size

    for cluster_set in cluster_sets:
        if len(train_indices) + len(cluster_set) <= train_cutoff:
            train_indices.extend(cluster_set)
        else:
            test_indices.extend(cluster_set)

    return train_indices, test_indices


def butina_split_from_csv(
    features,
    cluster_csv_path,
    frac_train=0.8,
    frac_test=0.2,
    frac_valid=0.0,
    cluster_column="Cluster_at_cutoff_0.3",
):
    """Split features by precomputed Butina cluster assignments from a CSV lookup."""
    assert abs(frac_train + frac_test + frac_valid - 1.0) < 1e-6

    total_size = len(features)
    if total_size == 0:
        if frac_valid > 0:
            return [], [], []
        return [], []

    if frac_valid == 0.0:
        train_size = int(frac_train * total_size)
        test_size = total_size - train_size
        train_indices, test_indices = _precomputed_butina_train_test_indices(
            features,
            train_size=train_size,
            test_size=test_size,
            cluster_csv_path=cluster_csv_path,
            cluster_column=cluster_column,
        )
        train_dataset = _subset_features(features, train_indices)
        test_dataset = _subset_features(features, test_indices)
        return train_dataset, test_dataset

    train_valid_size = total_size - int(frac_test * total_size)
    test_size = total_size - train_valid_size
    train_valid_indices, test_indices = _precomputed_butina_train_test_indices(
        features,
        train_size=train_valid_size,
        test_size=test_size,
        cluster_csv_path=cluster_csv_path,
        cluster_column=cluster_column,
    )

    train_valid_features = [features[i] for i in train_valid_indices]
    relative_train_frac = frac_train / (frac_train + frac_valid)
    train_size = int(relative_train_frac * len(train_valid_features))
    valid_size = len(train_valid_features) - train_size
    train_indices_within, valid_indices_within = _precomputed_butina_train_test_indices(
        train_valid_features,
        train_size=train_size,
        test_size=valid_size,
        cluster_csv_path=cluster_csv_path,
        cluster_column=cluster_column,
    )

    train_indices = [train_valid_indices[i] for i in train_indices_within]
    valid_indices = [train_valid_indices[i] for i in valid_indices_within]

    train_dataset = _subset_features(features, train_indices)
    test_dataset = _subset_features(features, test_indices)
    valid_dataset = _subset_features(features, valid_indices)

    return train_dataset, test_dataset, valid_dataset


def butina_split(
    features,
    frac_train=0.8,
    frac_test=0.2,
    frac_valid=0.0,
    cluster_csv_path=None,
    cluster_column="Cluster_at_cutoff_0.3",
):
    if cluster_csv_path is not None:
        print("Using csv file")
        return butina_split_from_csv(
            features,
            cluster_csv_path=cluster_csv_path,
            frac_train=frac_train,
            frac_test=frac_test,
            frac_valid=frac_valid,
            cluster_column=cluster_column,
        )

    if butina_train_test_split is None:
        raise ImportError(
            "skfp is required for butina_split() but is not installed in this environment."
        )

    assert abs(frac_train + frac_test + frac_valid - 1.0) < 1e-6

    total_size = len(features)
    if total_size == 0:
        if frac_valid > 0:
            return [], [], []
        return [], []

    if frac_valid == 0.0:
        train_size = int(frac_train * total_size)
        test_size = total_size - train_size
        train_indices, test_indices = _butina_train_test_indices(
            features,
            train_size=train_size,
            test_size=test_size,
        )
        train_dataset = _subset_features(features, train_indices)
        test_dataset = _subset_features(features, test_indices)
        return train_dataset, test_dataset

    train_valid_size = total_size - int(frac_test * total_size)
    test_size = total_size - train_valid_size
    train_valid_indices, test_indices = _butina_train_test_indices(
        features,
        train_size=train_valid_size,
        test_size=test_size,
    )

    train_valid_features = [features[i] for i in train_valid_indices]
    relative_train_frac = frac_train / (frac_train + frac_valid)
    train_size = int(relative_train_frac * len(train_valid_features))
    valid_size = len(train_valid_features) - train_size
    train_indices_within, valid_indices_within = _butina_train_test_indices(
        train_valid_features,
        train_size=train_size,
        test_size=valid_size,
    )

    train_indices = [train_valid_indices[i] for i in train_indices_within]
    valid_indices = [train_valid_indices[i] for i in valid_indices_within]

    train_dataset = _subset_features(features, train_indices)
    test_dataset = _subset_features(features, test_indices)
    valid_dataset = _subset_features(features, valid_indices)

    return train_dataset, test_dataset, valid_dataset
