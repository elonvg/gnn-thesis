from collections import Counter, defaultdict, deque
from functools import lru_cache
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

try:
    from skfp.model_selection import butina_train_test_split
except ImportError:
    butina_train_test_split = None


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


def _normalize_attribute_value(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Stratified splitting expects scalar graph attributes per item.")
        value = value.item()

    if isinstance(value, np.generic):
        value = value.item()

    if pd.isna(value):
        return None

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def _normalize_stratify_by(stratify_by):
    if stratify_by is None:
        return None

    if isinstance(stratify_by, str):
        attribute_names = (stratify_by,)
    else:
        attribute_names = tuple(stratify_by)

    if not attribute_names:
        raise ValueError("stratify_by must contain at least one attribute name.")

    for attribute_name in attribute_names:
        if not isinstance(attribute_name, str) or not attribute_name:
            raise ValueError("stratify_by must contain non-empty attribute names.")

    return attribute_names


def _feature_stratum_key(feature, attribute_names, feature_index):
    values = []

    for attribute_name in attribute_names:
        if not hasattr(feature, attribute_name):
            raise AttributeError(
                f"Feature at index {feature_index} is missing attribute {attribute_name!r}, "
                "which is required for stratified splitting."
            )

        values.append(_normalize_attribute_value(getattr(feature, attribute_name)))

    if len(values) == 1:
        return values[0]

    return tuple(values)


def _resolve_split_sizes(total_size, split_fractions):
    expected_sizes = {
        split_name: total_size * fraction
        for split_name, fraction in split_fractions.items()
    }
    split_sizes = {
        split_name: int(np.floor(expected_size))
        for split_name, expected_size in expected_sizes.items()
    }

    remaining = total_size - sum(split_sizes.values())
    if remaining <= 0:
        return split_sizes

    split_order = sorted(
        split_fractions,
        key=lambda split_name: (
            expected_sizes[split_name] - split_sizes[split_name],
            split_fractions[split_name],
        ),
        reverse=True,
    )

    for split_name in split_order[:remaining]:
        split_sizes[split_name] += 1

    return split_sizes


def _build_precomputed_cluster_records(
    features,
    cluster_csv_path,
    cluster_column="Cluster_at_cutoff_0.3",
    stratify_by=None,
):
    smiles_to_cluster = load_precomputed_butina_clusters(cluster_csv_path, cluster_column) # dict mapping SMILES to cluster IDs from the CSV file
    indices_by_cluster = defaultdict(list)
    strata_counts_by_cluster = defaultdict(Counter)
    total_strata_counts = Counter()

    for idx, feature in enumerate(features):
        cluster_key = _precomputed_cluster_key(feature.smiles, smiles_to_cluster)
        indices_by_cluster[cluster_key].append(idx)

        if stratify_by is not None:
            stratum_key = _feature_stratum_key(feature, stratify_by, idx)
            strata_counts_by_cluster[cluster_key][stratum_key] += 1
            total_strata_counts[stratum_key] += 1

    cluster_records = []
    for cluster_key, indices in indices_by_cluster.items():
        cluster_records.append(
            {
                "cluster_key": cluster_key,
                "indices": indices,
                "size": len(indices),
                "strata_counts": strata_counts_by_cluster.get(cluster_key, Counter()),
            }
        )

    return cluster_records, total_strata_counts


def _cluster_sort_key(cluster_record, total_strata_counts):
    if not cluster_record["strata_counts"]:
        return (-cluster_record["size"], cluster_record["indices"][0])

    rarest_label_count = min(
        total_strata_counts[stratum_key]
        for stratum_key in cluster_record["strata_counts"]
    )
    return (
        rarest_label_count,
        -cluster_record["size"],
        -len(cluster_record["strata_counts"]),
        cluster_record["indices"][0],
    )


def _assignment_score(
    split_name,
    cluster_record,
    split_sizes,
    split_strata_counts,
    target_sizes,
    target_strata_counts,
):
    target_size = max(target_sizes[split_name], 1)
    current_size = split_sizes[split_name]
    new_size = current_size + cluster_record["size"]

    current_size_gap = abs(current_size - target_sizes[split_name]) / target_size
    new_size_gap = abs(new_size - target_sizes[split_name]) / target_size
    current_overflow = max(0, current_size - target_sizes[split_name]) / target_size
    new_overflow = max(0, new_size - target_sizes[split_name]) / target_size

    current_counts = split_strata_counts[split_name]
    target_counts = target_strata_counts[split_name]

    current_strata_gap = sum(
        abs(current_counts.get(stratum_key, 0) - target_count)
        for stratum_key, target_count in target_counts.items()
    ) / target_size
    new_strata_gap = sum(
        abs(
            current_counts.get(stratum_key, 0)
            + cluster_record["strata_counts"].get(stratum_key, 0)
            - target_count
        )
        for stratum_key, target_count in target_counts.items()
    ) / target_size

    return (
        6.0 * (new_overflow - current_overflow)
        + 2.0 * (new_size_gap - current_size_gap)
        + (new_strata_gap - current_strata_gap)
    )


def _stratified_precomputed_butina_split_indices(
    features,
    split_fractions,
    cluster_csv_path,
    cluster_column="Cluster_at_cutoff_0.3",
    stratify_by=None,
):
    stratify_by = _normalize_stratify_by(stratify_by) # Get stratify_by in the right format (tuple of attribute names or None)
    cluster_records, total_strata_counts = _build_precomputed_cluster_records(
        features,
        cluster_csv_path=cluster_csv_path,
        cluster_column=cluster_column,
        stratify_by=stratify_by,
    )

    if not cluster_records:
        return {split_name: [] for split_name in split_fractions}

    cluster_records = sorted(
        cluster_records,
        key=lambda cluster_record: _cluster_sort_key(cluster_record, total_strata_counts),
    )

    target_sizes = _resolve_split_sizes(len(features), split_fractions)
    overall_strata_distribution = {
        stratum_key: count / len(features)
        for stratum_key, count in total_strata_counts.items()
    }
    target_strata_counts = {
        split_name: {
            stratum_key: target_sizes[split_name] * distribution
            for stratum_key, distribution in overall_strata_distribution.items()
        }
        for split_name in split_fractions
    }

    assigned_indices = {split_name: [] for split_name in split_fractions}
    split_sizes = {split_name: 0 for split_name in split_fractions}
    split_strata_counts = {split_name: Counter() for split_name in split_fractions}

    for cluster_record in cluster_records:
        candidate_order = sorted(
            split_fractions,
            key=lambda split_name: (
                _assignment_score(
                    split_name,
                    cluster_record,
                    split_sizes,
                    split_strata_counts,
                    target_sizes,
                    target_strata_counts,
                ),
                split_sizes[split_name] / max(target_sizes[split_name], 1),
                -target_sizes[split_name],
            ),
        )
        selected_split = candidate_order[0]

        assigned_indices[selected_split].extend(cluster_record["indices"])
        split_sizes[selected_split] += cluster_record["size"]
        split_strata_counts[selected_split].update(cluster_record["strata_counts"])

    return assigned_indices


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
    stratify_by=None,
):
    assert abs(frac_train + frac_test + frac_valid - 1.0) < 1e-6

    total_size = len(features)
    if total_size == 0:
        if frac_valid > 0:
            return [], [], []
        return [], []

    # Stratified splitting
    if stratify_by is not None:
        split_fractions = {"train": frac_train}
        if frac_valid > 0.0:
            split_fractions["valid"] = frac_valid
        split_fractions["test"] = frac_test

        split_indices = _stratified_precomputed_butina_split_indices(
            features,
            split_fractions=split_fractions,
            cluster_csv_path=cluster_csv_path,
            cluster_column=cluster_column,
            stratify_by=stratify_by,
        )

        train_dataset = _subset_features(features, split_indices["train"])
        test_dataset = _subset_features(features, split_indices["test"])
        if frac_valid == 0.0:
            return train_dataset, test_dataset

        valid_dataset = _subset_features(features, split_indices["valid"])
        return train_dataset, test_dataset, valid_dataset

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
    stratify_by=None,
):
    if cluster_csv_path is not None:
        print("Using csv file for butina split")
        return butina_split_from_csv(
            features,
            cluster_csv_path=cluster_csv_path,
            frac_train=frac_train,
            frac_test=frac_test,
            frac_valid=frac_valid,
            cluster_column=cluster_column,
            stratify_by=stratify_by,
        )

    if stratify_by is not None:
        raise ValueError(
            "stratify_by is only supported when using precomputed Butina clusters via cluster_csv_path."
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
