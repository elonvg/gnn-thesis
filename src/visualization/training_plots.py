import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def plot_toxicity_distribution(train_dataset, test_dataset, val_dataset=None):
    train_y = [g.y.item() for g in train_dataset]
    test_y = [g.y.item() for g in test_dataset]
    val_y = [g.y.item() for g in val_dataset] if val_dataset is not None else None

    plt.figure(figsize=(8, 4))
    plt.hist(train_y, bins=50, alpha=0.5, label="Train", density=True)
    if val_y is not None:
        plt.hist(val_y, bins=50, alpha=0.5, label="Val", density=True)
    plt.hist(test_y, bins=50, alpha=0.5, label="Test", density=True)
    plt.xlabel("log10c")
    plt.legend()
    title = "Target distribution: train vs val vs test" if val_y is not None else "Target distribution: train vs test"
    plt.title(title)
    plt.show()


def plot_training(history, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.plot(history["train_loss"], label="Train Loss")

    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="Val Loss")
    if history.get("test_loss"):
        plt.plot(history["test_loss"], label="Test Loss")

    best_epoch = history.get("best_epoch")
    if best_epoch is not None:
        plt.axvline(best_epoch, color="k", linestyle="--", alpha=0.4, label="Best Epoch")

    title = "Training Loss"
    if history.get("val_loss") and history.get("test_loss"):
        title = "Training, Validation, and Test Loss"
    elif history.get("val_loss"):
        title = "Training and Validation Loss"
    elif history.get("test_loss"):
        title = "Training and Test Loss"

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_training_metrics(history, metrics=None, prefixes=None, figsize=(15, 4)):
    metrics = list(metrics or ["loss", "rmse", "mae"])
    prefixes = list(prefixes or ["train", "val", "test"])

    available_metrics = []
    for metric in metrics:
        keys = [f"{prefix}_{metric}" for prefix in prefixes if history.get(f"{prefix}_{metric}")]
        if keys:
            available_metrics.append(metric)

    if not available_metrics:
        raise ValueError("No requested metrics were found in the training history.")

    fig, axes = plt.subplots(1, len(available_metrics), figsize=figsize, squeeze=False)

    for ax, metric in zip(axes[0], available_metrics):
        for prefix in prefixes:
            key = f"{prefix}_{metric}"
            values = history.get(key)
            if values:
                ax.plot(values, label=prefix.capitalize())

        best_epoch = history.get("best_epoch")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="k", linestyle="--", alpha=0.4)

        ax.set_title(metric.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    plt.show()


def build_label_decoders(label_encoders):
    if label_encoders is None:
        return {}

    return {
        category: {encoded: original for original, encoded in encoder.items()}
        for category, encoder in label_encoders.items()
    }


def _to_array(values):
    values = values or []
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)


def _best_value(values):
    values = _to_array(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(values.min())


def _final_value(values):
    values = _to_array(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(values[-1])


def _choose_summary_split(group_history, preferred=None):
    if preferred is not None and np.isfinite(_to_array(group_history.get(f"{preferred}_loss"))).any():
        return preferred

    for split in ("val", "test", "train"):
        if np.isfinite(_to_array(group_history.get(f"{split}_loss"))).any():
            return split

    return "train"


def _group_sample_count(group_history):
    return sum(int(group_history.get(f"{split}_n") or 0) for split in ("train", "val", "test"))


def _decode_group_label(category, group_value, label_decoders):
    label = label_decoders.get(category, {}).get(group_value, group_value)

    if label in {-1, None, "<NA>", "nan", "None"}:
        return "Missing"

    try:
        if np.isnan(label):
            return "Missing"
    except TypeError:
        pass

    return str(label)


def _group_summary_text(group_history, summary_split):
    counts = []
    for split in ("train", "val", "test"):
        n_samples = group_history.get(f"{split}_n")
        if n_samples is not None:
            counts.append(f"{split}={int(n_samples)}")

    best_loss = _best_value(group_history.get(f"{summary_split}_loss"))
    best_mae = _best_value(group_history.get(f"{summary_split}_mae"))
    final_loss = _final_value(group_history.get(f"{summary_split}_loss"))
    final_mae = _final_value(group_history.get(f"{summary_split}_mae"))

    lines = []
    if counts:
        lines.append("n: " + ", ".join(counts))
    lines.append(
        f"best {summary_split}: loss={best_loss:.3f}, mae={best_mae:.3f}"
    )
    lines.append(
        f"final {summary_split}: loss={final_loss:.3f}, mae={final_mae:.3f}"
    )

    return "\n".join(lines)


def _log_wandb_figure(run, key, fig):
    if run is None or wandb is None:
        return

    run.log({key: wandb.Image(fig)})


def plot_group_training(
    history,
    record_categories=None,
    metric="loss",
    top_n=5,
    summary_prefix=None,
    label_encoders=None,
    figsize=None,
    run=None,
):
    if metric not in {"loss", "rmse", "mae"}:
        raise ValueError("metric must be one of: 'loss', 'rmse', 'mae'.")
    categories = record_categories or [
        key.replace("history_", "", 1)
        for key in history
        if key.startswith("history_") and key != "history_all"
    ]
    label_decoders = build_label_decoders(label_encoders)
    best_epoch = history.get("history_all", {}).get("best_epoch")

    rows = []
    y_values = []
    for category in categories:
        group_history = history[f"history_{category}"][f"history_{category}_group"]
        groups = sorted(
            group_history.items(),
            key=lambda item: _group_sample_count(item[1]),
            reverse=True,
        )[:top_n]
        rows.append((category, groups))

        for _, values_by_group in groups:
            for split in ("train", "val", "test"):
                series = _to_array(values_by_group.get(f"{split}_{metric}"))
                y_values.extend(series[np.isfinite(series)])

    if figsize is None:
        figsize = (4.3 * top_n, 3.2 * len(rows))

    fig, axes = plt.subplots(len(rows), top_n, figsize=figsize, squeeze=False, sharex=True, sharey=True)
    legend_handles = {}

    for row_index, (category, groups) in enumerate(rows):
        for col_index in range(top_n):
            ax = axes[row_index, col_index]

            if col_index >= len(groups):
                ax.axis("off")
                continue

            group_value, group_history = groups[col_index]

            for split in ("train", "val", "test"):
                series = _to_array(group_history.get(f"{split}_{metric}"))
                if not np.isfinite(series).any():
                    continue

                epochs = np.arange(len(series))
                (line,) = ax.plot(epochs, series, linewidth=1.8, label=split.capitalize())
                legend_handles.setdefault(split, line)

            if best_epoch is not None:
                ax.axvline(best_epoch, color="k", linestyle="--", alpha=0.35)

            display_label = _decode_group_label(category, group_value, label_decoders)
            total_n = _group_sample_count(group_history)
            ax.set_title(f"{display_label} (n={total_n})", fontsize=10)
            ax.grid(alpha=0.3)

            if col_index == 0:
                ax.set_ylabel(f"{category}\n{metric.upper()}")
            if row_index == len(rows) - 1:
                ax.set_xlabel("Epoch")

            summary_split = _choose_summary_split(group_history, preferred=summary_prefix)

    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        padding = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
        for ax in axes.flat:
            if ax.has_data():
                ax.set_ylim(y_min - padding, y_max + padding)

    if len(categories) == 1:
        fig.suptitle(f"Training {metric.upper()} by {categories[0]}", y=0.995)
    else:
        fig.suptitle(f"Grouped Training {metric.upper()} by Category", y=0.995)

    if legend_handles:
        fig.legend(
            [legend_handles[split] for split in legend_handles],
            [split.capitalize() for split in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=len(legend_handles),
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        fig.tight_layout()

    if len(categories) == 1:
        log_key = f"categories/{categories[0]}/training_{metric}"
    else:
        log_key = f"categories/grouped_training_{metric}"

    _log_wandb_figure(run, log_key, fig)
    plt.show()
    return fig
