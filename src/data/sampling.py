from collections import Counter, defaultdict, deque
from functools import lru_cache
from copy import deepcopy
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

import numpy as np
import pandas as pd
import torch

def _normalize_attribute_value(value):
    # Function for normalizing attribute values

    # Convert tensor to scalar if it's a single-element tensor
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Weighted sampling expects scalar graph attributes per item.")
        value = value.item()

    # Convert numpy scalar to native Python scalar
    if isinstance(value, np.generic):
        value = value.item()

    # Treat NaN as None
    if pd.isna(value):
        return None

    # Convert floats that are mathematically integers to int type for cleaner attribute values
    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def collect_attribute_values(dataset, attribute_name):
    # Function for collecting values of an attribute over a dataset

    values = []

    for idx, item in enumerate(dataset):
        if not hasattr(item, attribute_name):
            raise AttributeError(
                f"Dataset item at index {idx} is missing attribute {attribute_name!r}, "
                "which is required for weighted sampling."
            )

        values.append(_normalize_attribute_value(getattr(item, attribute_name)))

    return values


def compute_attribute_distribution(dataset, attribute_name):
    # Function for computing distribution of an attribute in the dataset
    # Returns a dict mapping attribute values to number of occurances
    
    # Collect attribute values
    values = collect_attribute_values(dataset, attribute_name)
    if not values:
        return {}

    # Compute value counts and total
    counts = Counter(values)
    total = float(len(values))
    return {label: count for label, count in counts.items()}


def build_attribute_weights(dataset, attribute_name, target_distribution=None):
    # Function for building sampling weights for dataset

    values = collect_attribute_values(dataset, attribute_name)
    if not values:
        raise ValueError("Weighted sampling requires a non-empty dataset.")

    counts = Counter(values)

    if target_distribution is None:
        target_distribution = {
            label: count / len(values)
            for label, count in counts.items()
        }
    else:
        normalized_target = {}
        for label, weight in target_distribution.items():
            normalized_label = _normalize_attribute_value(label)
            normalized_weight = float(weight)
            if normalized_weight < 0:
                raise ValueError("target_distribution weights must be non-negative.")
            normalized_target[normalized_label] = normalized_weight
        target_distribution = normalized_target

    present_target_mass = sum(target_distribution.get(label, 0.0) for label in counts)
    if present_target_mass <= 0:
        raise ValueError(
            f"target_distribution does not assign any positive mass to labels present in "
            f"attribute {attribute_name!r}."
        )

    normalized_target = {
        label: target_distribution.get(label, 0.0) / present_target_mass
        for label in counts
    }

    return torch.tensor(
        [normalized_target[label] / counts[label] for label in values],
        dtype=torch.double,
    )

def simple_weights(dataset, attribute_distribution):
    weights = []
    total = len(dataset)

    for item, n in attribute_distribution:
        weights.append(n/total)

def build_weighted_random_sampler(
    dataset,
    attribute_name,
    target_distribution=None,
    num_samples=None,
    replacement=True,
):
    # Function for building a WeightedRandomSampler for a dataset based on an attribute distribution

    weights = build_attribute_weights(
        dataset,
        attribute_name=attribute_name,
        target_distribution=target_distribution,
    )
    if num_samples is None:
        num_samples = len(weights)

    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=replacement,
    )

def LoadData(dataset, batch_size, shuffle, weighted_random_sampler, attribute):

    # Compute number of samples for each group
    attr_dist = compute_attribute_distribution(dataset, attribute)

    # Compute weights : 1/count
    weights = simple_weights(dataset, attr_dist)
    # Create sampler

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=None,
        replacement=True
        )

    # Create dataloader
    loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    sampler=sampler
    )

    return loader