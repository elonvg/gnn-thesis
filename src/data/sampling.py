from collections import Counter
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

def compute_attribute_distribution(values, attribute_name=None):
    # Function for computing distribution of an attribute in the dataset
    # Returns a dict mapping attribute values to number of occurances
    if attribute_name is not None:
        values = collect_attribute_values(values, attribute_name)
    
    if not values:
        return {}

    # Compute value counts and total
    counts = Counter(values)
    total = float(len(values))
    return {label: count/total for label, count in counts.items()}

def compute_weights(target_distribution, values):
    # Count frequency of each value
    counts = Counter(values)

    # Normalize values
    normalized_target = {}
    for value, weight in target_distribution.items():
        normalized_value = _normalize_attribute_value(value)
        normalized_weight = float(weight)

        normalized_target[normalized_value] = normalized_weight
    target_distribution = normalized_target
    
    # Normalize target values over labels present in this dataset.
    target_mass = sum(target_distribution.get(value, 0.0) for value in counts)
    if target_mass <= 0:
        raise ValueError(
            "Target distribution does not contain any labels present in the dataset."
        )

    normalized_target = {
        value: target_distribution.get(value, 0.0) / target_mass for value in counts
    }

    # weight = target / total
    # weights = torch.tensor(
    #     [normalized_target[value] / counts[value] for value in values], dtype=torch.double
    # )

    weights = torch.tensor(
        [1 / counts[value] for value in values], dtype=torch.double
    )

    return weights


def LoadData(dataset, batch_size, shuffle=False, attribute="species_group", target_dataset=None):
    if target_dataset is None:
        target_dataset = dataset
    
    # Compute target distribution from target dataset
    target_values = collect_attribute_values(target_dataset, attribute)

    target_attr_dist = compute_attribute_distribution(target_values)

    # Collect values (list of specified attribute for each point in dataset)
    values = collect_attribute_values(dataset, attribute)

    # Compute weights : target/count
    weights = compute_weights(target_attr_dist, values)

    # Create sampler
    num_samples = len(dataset)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
    )

    return loader


def _loader_attribute_counts(dataloader, attribute):
    counts = Counter()

    for batch in dataloader:
        try:
            values = batch[attribute]
        except (KeyError, TypeError):
            values = getattr(batch, attribute)

        if torch.is_tensor(values):
            values = values.detach().cpu().tolist()
        elif hasattr(values, "tolist"):
            values = values.tolist()

        if not isinstance(values, (list, tuple)):
            values = [values]

        counts.update(_normalize_attribute_value(value) for value in values)

    return counts


def _normalize_dataloaders_for_plot(dataloaders):
    if isinstance(dataloaders, dict):
        return list(dataloaders.items())

    if isinstance(dataloaders, (list, tuple)):
        if dataloaders and isinstance(dataloaders[0], tuple) and len(dataloaders[0]) == 2:
            return list(dataloaders)

        default_names = ("Train", "Val", "Test")
        return [
            (default_names[index] if index < len(default_names) else f"Loader {index + 1}", dataloader)
            for index, dataloader in enumerate(dataloaders)
        ]

    return [("Loader", dataloaders)]


def _decode_attribute_label(value, species_group_decoder=None):
    if species_group_decoder is None:
        return str(value)

    if isinstance(species_group_decoder, dict):
        value = species_group_decoder.get(value, value)

    if value in {-1, None, "<NA>", "nan", "None"}:
        return "Missing"

    try:
        if np.isnan(value):
            return "Missing"
    except TypeError:
        pass

    return str(value)


def display_dataloader_distribution(dataloaders, attribute, species_group_decoder=None, figsize=None):
    import matplotlib.pyplot as plt

    named_loaders = _normalize_dataloaders_for_plot(dataloaders)
    counts_by_loader = [
        (loader_name, _loader_attribute_counts(dataloader, attribute))
        for loader_name, dataloader in named_loaders
    ]

    labels = sorted(
        {label for _, counts in counts_by_loader for label in counts},
        key=lambda value: str(value),
    )
    if not labels:
        raise ValueError("No attribute values were found in the dataloader(s).")

    if figsize is None:
        figsize = (max(8, 0.65 * len(labels) + 3), 5)

    x = np.arange(len(labels))
    width = min(0.8 / len(counts_by_loader), 0.25)

    fig, ax = plt.subplots(figsize=figsize)
    for index, (loader_name, counts) in enumerate(counts_by_loader):
        total = sum(counts.values())
        frequencies = [counts[label] / total if total else 0.0 for label in labels]
        offset = (index - (len(counts_by_loader) - 1) / 2) * width
        bars = ax.bar(x + offset, frequencies, width=width, label=loader_name)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    tick_labels = [
        _decode_attribute_label(label, species_group_decoder)
        for label in labels
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    ax.set_xlabel(attribute)
    ax.set_ylabel("Fraction")
    ax.set_title(f"{attribute} distribution by dataloader")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()


def show_loader_info(attribute, train_loader, val_loader, test_loader, species_group_decoder):
    loaders = {
        "Train": train_loader,
        "Val": val_loader,
        "Test": test_loader,
    }

    for loader_name, loader in loaders.items():
        dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else "unknown"
        sampled_size = getattr(getattr(loader, "sampler", None), "num_samples", dataset_size)
        batch_size = getattr(loader, "batch_size", "unknown")
        print(
            f"{loader_name}: {dataset_size} dataset samples, "
            f"{sampled_size} sampled samples, {len(loader)} batches "
            f"(batch_size={batch_size})"
        )

    display_dataloader_distribution(loaders, attribute, species_group_decoder=species_group_decoder)

def display_sampling_effect(
    dataset,
    loader,
    attribute,
    categorical_decoder=None,
    figsize=None,
):
    """
    Side-by-side: raw dataset distribution vs what the
    weighted sampler actually draws across all batches.
    Optionally adds a third panel showing the per-sample weights.
    """
    import matplotlib.pyplot as plt

    # --- raw dataset distribution ---
    raw_values = collect_attribute_values(dataset, attribute)
    raw_dist = compute_attribute_distribution(raw_values)

    # --- sampled distribution (iterate loader) ---
    sampled_counts = _loader_attribute_counts(loader, attribute)
    sampled_total = sum(sampled_counts.values())
    sampled_dist = {
        k: v / sampled_total for k, v in sampled_counts.items()
    }

    # --- align labels ---
    labels = sorted(set(raw_dist) | set(sampled_dist), key=str)
    tick_labels = [
        _decode_attribute_label(l, categorical_decoder) for l in labels
    ]

    x = np.arange(len(labels))
    w = 0.35

    if figsize is None:
        figsize = (max(8, 0.7 * len(labels) + 3), 5)

    fig, ax = plt.subplots(figsize=figsize)

    raw_vals  = [raw_dist.get(l, 0.0) for l in labels]
    samp_vals = [sampled_dist.get(l, 0.0) for l in labels]

    bars_raw  = ax.bar(x - w/2, raw_vals,  w, label="Dataset (raw)",   color="#85B7EB")
    bars_samp = ax.bar(x + w/2, samp_vals, w, label="Sampled (loader)", color="#5DCAA5")

    ax.bar_label(bars_raw,  fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(bars_samp, fmt="%.2f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    ax.set_xlabel(attribute)
    ax.set_ylabel("Fraction")
    ax.set_title(f"Effect of weighted sampling on {attribute} distribution")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()