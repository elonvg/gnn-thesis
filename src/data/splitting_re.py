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

@lru_cache(maxsize=None)
def load_butina_clusters(cluster_csv_path, cluster_col):
    # Function for loading precomputer butina cluster assignments
    # Returns a dict mapping SMILES to cluster IDs from the CSV file

    cluster_df = pd.read_csv(
        cluster_csv_path,
        usecols=["SMILES", cluster_col],
        compression="infer",
        low_memory=False,
        )
    
    return dict(zip(cluster_df["SMILES"], cluster_df[cluster_col]))

def _precomputed_cluster_key(smiles, smiles_to_cluster):
    # Helper function to get the cluster key for a given SMILES string based on the precomputed clusters
    # Returns a string key in the format "cluster::{cluster_id}" or "__missing__::{smiles}" if not found

    cluster_id = smiles_to_cluster.get(smiles, np.nan)
    if pd.isna(cluster_id):
        return f"__missing__::{smiles}"
    
    return f"cluster::{cluster_id}"

def _butina_train_test_indices(
        features,
        train_size,
        test_size,
        cluster_csv_path,
        cluster_col="Cluster_at_cutoff_0.3",
):
    # Function for computing indices for train/test based on precomputed butina cluster assignments
    # Returns two lists of indices for train/test splits

    smiles_to_cluster = load_butina_clusters(cluster_csv_path, cluster_col) # Dict mapping SMILES to cluster IDs
    indices_by_cluster = defaultdict(list) # Store indices of features by their cluster key

    for idx, feature in enumerate(features):
        cluster_key = _precomputed_cluster_key(feature.smiles, smiles_to_cluster)
        indices_by_cluster[cluster_key].append(idx)

    cluster_sets = sorted(indices_by_cluster.values(), key=lambda indices: (-len(indices), indices[0])) # Sort clusters by size (descending) and then by first index (ascending)

    train_indices, test_indices = [], []
    train_cutoff = train_size

    # Add entire clusters to train set until we reach the train cutoff, then add remaining clusters to test set
    for cluster_set in cluster_sets:
        if len(train_indices) + len(cluster_set) <= train_cutoff:
            train_indices.extend(cluster_set)
        else:
            test_indices.extend(cluster_set)

    return train_indices, test_indices


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


def _feature_stratum_key(feature, stratify_by, feature_index):
    # Function for computing the straification group for one feature

    values = []

    for attribute in stratify_by:
        if hasattr(feature, attribute):
            value = getattr(feature, attribute)
            values.append(_normalize_attribute_value(value))
        else:
            raise ValueError(f"Feature at index {feature_index} does not have attribute '{attribute}' for stratification.")
        
    if len(values) == 1:
        return values[0]
    
    return tuple(values)

def _build_cluster_records(
        features,
        stratify_by,
        cluster_csv_path,
        cluster_col="Cluster_at_cutoff_0.3",
):
    # Function for building per-cluster summaries of indices, size and strata_counts

    smiles_to_cluster = load_butina_clusters(cluster_csv_path, cluster_col) # Dict mapping SMILES to cluster IDs from the CSV file
    indices_by_cluster = defaultdict(list) # Store indices of features by their cluster key
    strata_counts_by_cluster = defaultdict(Counter) # Store counts of stratification values by cluster key
    total_strata_counts = Counter()

    for i, feature in enumerate(features):
        cluster_key = _precomputed_cluster_key(feature.smiles, smiles_to_cluster)
        indices_by_cluster[cluster_key].append(i)

        if stratify_by is not None:
            stratum_key = _feature_stratum_key(feature, stratify_by, i)
            strata_counts_by_cluster[cluster_key][stratum_key] += 1
            total_strata_counts[stratum_key] += 1

    cluster_records = []
    for cluster_key, indices, in indices_by_cluster.items():
        cluster_records.append(
            {
                "cluster_key": cluster_key,
                "indices": indices,
                "size": len(indices),
                "strata_counts": strata_counts_by_cluster[cluster_key],
            }
        )

    return cluster_records, total_strata_counts

def _cluster_sort_key(cluster_record, total_strata_counts):
    # Function for deciding which clusters shoud be sorted at the top
    # Priority: Rare strata, large clusters, more diverse strata
    # "Handles the most “difficult-to-place” clusters first so the greedy assignment doesn’t paint itself into a corner"

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

def _assignment_score(
    split_name,
    cluster_record,
    split_sizes,
    split_strata_counts,
    target_sizes,
    target_strata_counts,
):
    # Function for measuring how good it would be to place a cluster in a split
    # Lower score means better assignment (more likely to be selected)

    target_size = max(target_sizes[split_name], 1)
    curr_size = split_sizes[split_name]
    new_size = curr_size + cluster_record["size"]

    current_size_gap = abs(curr_size - target_sizes[split_name]) / target_size
    new_size_gap = abs(new_size - target_sizes[split_name]) / target_size
    current_overflow = max(0, curr_size - target_sizes[split_name]) / target_size
    new_overflow = max(0, new_size - target_sizes[split_name]) / target_size

    curr_counts = split_strata_counts[split_name]
    target_counts = target_strata_counts[split_name]

    current_strata_gap = sum(
        abs(curr_counts.get(stratum_key, 0) - target_count)
        for stratum_key, target_count in target_counts.items()
    ) / target_size
    new_strata_gap = sum(
        abs(
            curr_counts.get(stratum_key, 0)
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



def _stratified_butina_split_indices(
    features,
    stratify_by,
    split_fractions,
    cluster_csv_path,
    cluster_col="Cluster_at_cutoff_0.3",
):
    # Function for getting the final split indices from stratified butina splitting
    # Keeps clusters intact and sorts them into splits based on strata distribution and target split sizes

    stratify_by = _normalize_stratify_by(stratify_by) # Normalizes stratify_by to a tuple of attribute names
    
    # Get cluster_records with info about indices, size and strata counts for each cluster, as well as total strata counts across all clusters
    cluster_records, total_strata_counts = _build_cluster_records(
        features,
        stratify_by,
        cluster_csv_path,
        cluster_col,
    )

    if not cluster_records:
        return {split_name: [] for split_name in split_fractions}

    # Sort clusters by strata info
    cluster_records = sorted(
        cluster_records,
        key=lambda cluster_record: _cluster_sort_key(cluster_record, total_strata_counts)
    )    

    target_sizes = _resolve_split_sizes(len(features), split_fractions) # Fractions -> integers

    # Compute overall strata distribution
    overall_strata_dist = {
        stratum_key: count / len(features)
        for stratum_key, count in total_strata_counts.items()
    }
    # Compute ideal target strata distribution
    target_strata_counts = {
        split_name: {
            stratum_key: target_sizes[split_name] * dist
            for stratum_key, dist in overall_strata_dist.items()
        }
        for split_name in split_fractions
    }

    assigned_indices = {split_name: [] for split_name in split_fractions} # Store assigned indices for each split
    split_sizes = {split_name: 0 for split_name in split_fractions} # Store current sizes of each split
    split_strata_counts = {split_name: Counter() for split_name in split_fractions} # Store current strata counts for each split

    # Loop through clusters and assign them to splits in a greedy way, trying to maintain the target split sizes and strata distributions as much as possible
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



def _build_dataset(features, indices):
    # Function for building the subset (train/test) from given indices
    return [deepcopy(features[i]) for i in indices]

def butina_split_from_csv(
    features,
    stratify_by=None,
    frac_train=0.7,
    frac_valid=0.1,
    frac_test=0.2,
    cluster_csv_path=None,
    cluster_col="Cluster_at_cutoff_0.3",
):
    assert abs(frac_train + frac_test + frac_valid - 1.0) < 1e-6 # Check fractions

    total_size = len(features)
    train_size = int(frac_train * total_size)
    test_size = int(frac_test * total_size)
    val_size = total_size - train_size - test_size

    if total_size == 0:
        if frac_valid > 0:
            return [], [], []
        return [], []
    
    # Stratified splitting
    if stratify_by is not None:
        print("Using stratified splitting with stratify_by:", stratify_by)

        split_fractions = {}
        split_fractions["train"] = frac_train
        split_fractions["val"] = frac_valid
        split_fractions["test"] = frac_test

        split_indices = _stratified_butina_split_indices(
            features,
            stratify_by=stratify_by,
            split_fractions=split_fractions,
            cluster_csv_path=cluster_csv_path,
            cluster_col=cluster_col,
        )

        train_dataset = _build_dataset(features, split_indices["train"])
        val_dataset = _build_dataset(features, split_indices["val"])
        test_dataset = _build_dataset(features, split_indices["test"])

        return train_dataset, val_dataset, test_dataset

    # Non-stratified splitting
    train_val_indices, test_indices = _butina_train_test_indices(
        features,
        train_size = train_size + val_size,
        test_size = test_size,
    )

    train_valid_features = [features[i] for i in train_val_indices]
    relative_train_frac = frac_train / (frac_train + frac_valid)

    train_size = int(relative_train_frac * len(train_val_indices))
    valid_size = len(train_val_indices) - train_size

    train_indices, val_indices = _butina_train_test_indices(
        train_valid_features,
        train_size=train_size,
        test_size=valid_size,
    )

    train_indices = [train_val_indices[i] for i in train_indices]
    val_indices = [train_val_indices[i] for i in val_indices]

    train_dataset = _build_dataset(features, train_indices)
    val_dataset = _build_dataset(features, val_indices)
    test_dataset = _build_dataset(features, test_indices)

    return train_dataset, val_dataset, test_dataset

def butina_split(
    features,
    stratify_by=None,
    frac_train=0.7,
    frac_valid=0.1,
    frac_test=0.2,
    cluster_csv_path=None,
    cluster_col="Cluster_at_cutoff_0.3",
):
    if cluster_csv_path is None:
        raise ValueError("cluster_csv_path must be provided for butina_split")
    
    print("Using csv file for butina splitting:", cluster_csv_path)

    return butina_split_from_csv(
        features,
        stratify_by=stratify_by,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        cluster_csv_path=cluster_csv_path,
        cluster_col=cluster_col,
    )