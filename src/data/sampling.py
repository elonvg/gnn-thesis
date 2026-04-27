from collections import Counter, defaultdict, deque
from functools import lru_cache
from copy import deepcopy
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import pandas as pd

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

def compute_attribute_distribution(values):
    # Function for computing distribution of an attribute in the dataset
    # Returns a dict mapping attribute values to number of occurances
    
    if not values:
        return {}

    # Compute value counts and total
    counts = Counter(values)
    total = float(len(values))
    return {label: count/total for label, count in counts.items()}

def simple_weights(target_distribution, values):
    weights = []

    for n in target_distribution.values():
        weights.append(n)
    
    # Count frequency of each value
    counts = Counter(values)

    # Normalize values
    normalized_target = {}
    for value, weight in target_distribution.items():
        normalized_value = _normalize_attribute_value(value)
        normalized_weight = float(weight)

        normalized_target[normalized_value] = normalized_weight
    target_distribution = normalized_target
    
    # Normalze value values
    target_mass = sum(target_distribution.get(value, 0.0) for value in counts)
    normalized_target = {
        value: target_distribution.get(value, 0.0) / target_mass for value in counts
    }

    # weight = target / total
    weights = torch.tensor(
        [normalized_target[value] / counts[value] for value in values], dtype=torch.double
    )

    return weights

def LoadData(dataset, batch_size, shuffle=False, attribute="species_group", target_dataset=None):
    if target_dataset == None:
        target_dataset = dataset
    
    # Collect values (list of specified attribute for each point in dataset)
    values = collect_attribute_values(dataset, attribute)

    # Compute number of samples for each group
    target_attr_dist = compute_attribute_distribution(values)

    # Compute weights : 1/count
    weights = simple_weights(target_attr_dist, values)

    # Create sampler
    num_samples = len(dataset)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
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

def display_dataloader_distribution(dataloader, attribute):
    from collections import Counter
    import matplotlib.pyplot as plt

    counts = Counter()
    for batch in dataloader:
        values = batch[attribute]
        # handle tensors or plain lists
        if hasattr(values, "tolist"):
            values = values.tolist()
        counts.update(values)

    total = sum(counts.values())
    labels = sorted(counts.keys())
    frequencies = [counts[l] / total for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(l) for l in labels], frequencies)
    ax.set_xlabel(attribute)
    ax.set_ylabel("Fracrion")
    ax.set_title(f"Distribution of {attribute}")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    plt.tight_layout()
    plt.show()


def show_loader_info(attribute, train_loader, val_loader, test_loader, categorical_decoder):

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    display_dataloader_distribution(train_loader, attribute)