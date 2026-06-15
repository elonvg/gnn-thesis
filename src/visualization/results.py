import matplotlib.pyplot as plt
import numpy as np


RESULT_TARGET_COL = "actual_log10c"
PREDICTION_COL = "pred_log10c"
TRAIN_TARGET_CANDIDATES = ("actual_log10c", "log10c")


def _require_columns(frame, columns, frame_name):
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"{frame_name} is missing required column(s): {missing_str}")


def _normalize_group_labels(series):
    labels = series.astype("string")
    labels = labels.fillna("Missing").replace({"<NA>": "Missing", "nan": "Missing", "None": "Missing"})
    return labels.astype(str)


def _resolve_train_target_col(train_df):
    for candidate in TRAIN_TARGET_CANDIDATES:
        if candidate in train_df.columns:
            return candidate

    expected = ", ".join(TRAIN_TARGET_CANDIDATES)
    raise KeyError(f"train_df must include one of the training target columns: {expected}")


def summarize_by_group(results_df, train_df, group_col, min_count=20):
    _require_columns(results_df, [group_col, RESULT_TARGET_COL, PREDICTION_COL], "results_df")
    _require_columns(train_df, [group_col], "train_df")

    train_target_col = _resolve_train_target_col(train_df)

    df = results_df.copy()
    df = df.dropna(subset=[RESULT_TARGET_COL, PREDICTION_COL]).copy()
    df["group"] = _normalize_group_labels(df[group_col])

    train_groups = train_df.copy()
    train_groups = train_groups.dropna(subset=[train_target_col]).copy()
    train_groups["group"] = _normalize_group_labels(train_groups[group_col])

    global_mean = train_groups[train_target_col].mean()
    if np.isnan(global_mean):
        raise ValueError("train_df does not contain any non-null training target values.")

    baseline_stats = (
        train_groups.groupby("group")[train_target_col]
        .agg(train_n="size", baseline_log10c="mean")
        .reset_index()
    )

    df = df.merge(baseline_stats, on="group", how="left")
    df["baseline_log10c"] = df["baseline_log10c"].fillna(global_mean)
    df["train_n"] = df["train_n"].fillna(0).astype(int)
    df["baseline_source"] = np.where(df["train_n"] > 0, "train_group_mean", "global_train_mean")

    df["model_abs_error"] = (df[PREDICTION_COL] - df[RESULT_TARGET_COL]).abs()
    df["baseline_abs_error"] = (df["baseline_log10c"] - df[RESULT_TARGET_COL]).abs()
    df["model_sq_error"] = (df[PREDICTION_COL] - df[RESULT_TARGET_COL]) ** 2
    df["baseline_sq_error"] = (df["baseline_log10c"] - df[RESULT_TARGET_COL]) ** 2

    summary = (
        df.groupby("group")
        .agg(
            n=("group", "size"),
            train_n=("train_n", "first"),
            baseline_source=("baseline_source", "first"),
            baseline_log10c=("baseline_log10c", "first"),
            model_mae=("model_abs_error", "mean"),
            baseline_mae=("baseline_abs_error", "mean"),
            model_rmse=("model_sq_error", lambda x: np.sqrt(x.mean())),
            baseline_rmse=("baseline_sq_error", lambda x: np.sqrt(x.mean())),
        )
        .reset_index()
    )

    summary["mae_gain"] = summary["baseline_mae"] - summary["model_mae"]
    summary["rmse_gain"] = summary["baseline_rmse"] - summary["model_rmse"]

    if min_count is not None:
        summary = summary[summary["n"] >= min_count]

    return summary.sort_values(["n", "mae_gain"], ascending=[False, False]).reset_index(drop=True)


def plot_group_mae(summary, category, top_n=10):
    title = f"{category}: model vs train subgroup mean"

    if summary.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No groups to plot", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        return fig

    plot_df = summary.head(top_n).iloc[::-1]
    labels = [f"{group} (n={n})" for group, n in zip(plot_df["group"], plot_df["n"])]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, plot_df["baseline_mae"], color="#c7ced6", label="Train subgroup mean")
    ax.barh(labels, plot_df["model_mae"], color="#2f6db3", label="Model")
    ax.set_xlabel("MAE")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


from collections.abc import Mapping
import pandas as pd

def plot_median_abs_error_by_category(
    dfs,
    category_cols,
    *,
    error_col="abs_error_log10c",
    filters=None,
    labels=None,
    top_n=None,
    min_count=1,
    sort_by="median",
    ascending=False,
    figsize=None,
    rotation=45,
):
    """
    Plot median absolute error grouped by one or more categorical columns.

    dfs:
        Either a dict like {"AFP": afp_df, "GCN": gcn_df}
        or a list like [afp_df, gcn_df].

    category_cols:
        One column name or a list of column names to group by.

    filters:
        Optional dict to filter rows before grouping.
        Example: {"endpoint": "EC50", "species_group": ["fish", "rodents"]}
    """
    if isinstance(category_cols, str):
        category_cols = [category_cols]

    if isinstance(dfs, pd.DataFrame):
        dfs = {"df": dfs}
    elif isinstance(dfs, Mapping):
        dfs = dict(dfs)
    else:
        if labels is None:
            labels = [f"df_{i + 1}" for i in range(len(dfs))]
        dfs = dict(zip(labels, dfs))

    filters = filters or {}
    summaries = []

    for name, df in dfs.items():
        required_cols = set(category_cols + [error_col] + list(filters))
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise KeyError(f"{name} is missing columns: {sorted(missing_cols)}")

        filtered = df.copy()

        for col, allowed_values in filters.items():
            if not isinstance(allowed_values, (list, tuple, set, pd.Index)):
                allowed_values = [allowed_values]
            filtered = filtered[filtered[col].isin(allowed_values)]

        summary = (
            filtered
            .dropna(subset=category_cols + [error_col])
            .groupby(category_cols, dropna=False)[error_col]
            .agg(median_abs_error="median", n="size")
            .reset_index()
        )

        summary = summary[summary["n"] >= min_count]
        summary["df"] = name
        summary["category"] = summary[category_cols].astype(str).agg(" | ".join, axis=1)

        summaries.append(summary)

    summary_df = pd.concat(summaries, ignore_index=True)

    if summary_df.empty:
        raise ValueError("No rows left after filtering/grouping.")

    if sort_by == "median":
        order = summary_df.groupby("category")["median_abs_error"].median()
    elif sort_by == "count":
        order = summary_df.groupby("category")["n"].sum()
    else:
        raise ValueError("sort_by must be 'median' or 'count'")

    order = order.sort_values(ascending=ascending).index

    if top_n is not None:
        order = order[:top_n]

    plot_df = (
        summary_df[summary_df["category"].isin(order)]
        .pivot(index="category", columns="df", values="median_abs_error")
        .reindex(order)
    )

    if figsize is None:
        figsize = (max(8, len(plot_df) * 0.7), 5)

    ax = plot_df.plot(kind="bar", figsize=figsize)

    ax.set_xlabel(" | ".join(category_cols))
    ax.set_ylabel("Median absolute error")
    ax.set_title(f"Median absolute error by {' + '.join(category_cols)}")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=rotation)
    ax.legend(title="Dataframe")

    plt.tight_layout()

    return summary_df, ax

def plot_median_ae_by_fragment_count(
    dfs,
    *,
    fragment_col="fragment_count",
    error_col="abs_error_log10c",
    labels=None,
    max_fragment_count=None,
    min_count=1,
    figsize=(8, 5),
):
    if isinstance(dfs, pd.DataFrame):
        dfs = {"df": dfs}
    elif isinstance(dfs, Mapping):
        dfs = dict(dfs)
    else:
        if labels is None:
            labels = [f"df_{i + 1}" for i in range(len(dfs))]
        dfs = dict(zip(labels, dfs))

    summaries = []

    for name, df in dfs.items():
        work = df[[fragment_col, error_col]].copy()
        work[fragment_col] = pd.to_numeric(work[fragment_col], errors="coerce")
        work[error_col] = pd.to_numeric(work[error_col], errors="coerce")
        work = work.dropna()

        work[fragment_col] = work[fragment_col].astype(int)

        if max_fragment_count is not None:
            work = work[work[fragment_col] <= max_fragment_count]

        summary = (
            work
            .groupby(fragment_col)[error_col]
            .agg(median_ae="median", n="size")
            .reset_index()
        )

        summary = summary[summary["n"] >= min_count]
        summary["df"] = name
        summaries.append(summary)

    summary_df = pd.concat(summaries, ignore_index=True)

    fig, ax = plt.subplots(figsize=figsize)

    for name, group in summary_df.groupby("df"):
        group = group.sort_values(fragment_col)
        ax.plot(
            group[fragment_col],
            group["median_ae"],
            marker="o",
            linewidth=2,
            label=name,
        )

    ax.set_xlabel("Fragment count")
    ax.set_ylabel("Median absolute error")
    ax.set_title("Median absolute error by fragment count")
    ax.grid(alpha=0.3)
    ax.legend(title="Model")

    plt.tight_layout()

    return summary_df, ax

def plot_median_ae_by_column(
    dfs,
    column,
    *,
    error_col="abs_error_log10c",
    labels=None,
    bins=12,
    binning="quantile",
    max_exact_values=20,
    min_count=1,
    figsize=(9, 5),
):
    if isinstance(dfs, pd.DataFrame):
        dfs = {"df": dfs}
    elif isinstance(dfs, Mapping):
        dfs = dict(dfs)
    else:
        if labels is None:
            labels = [f"df_{i + 1}" for i in range(len(dfs))]
        dfs = dict(zip(labels, dfs))

    summaries = []

    for name, df in dfs.items():
        work = df[[column, error_col]].copy()
        work[error_col] = pd.to_numeric(work[error_col], errors="coerce")
        work[column] = pd.to_numeric(work[column], errors="ignore")
        work = work.dropna(subset=[column, error_col])

        is_numeric = pd.api.types.is_numeric_dtype(work[column])
        use_bins = is_numeric and work[column].nunique() > max_exact_values

        if use_bins:
            if binning == "quantile":
                work["_bin"] = pd.qcut(work[column], q=bins, duplicates="drop")
            elif binning == "equal_width":
                work["_bin"] = pd.cut(work[column], bins=bins)
            else:
                raise ValueError("binning must be 'quantile' or 'equal_width'")

            summary = (
                work.groupby("_bin", observed=True)[error_col]
                .agg(median_ae="median", n="size")
                .reset_index()
            )

            summary = summary[summary["n"] >= min_count]
            summary["x"] = summary["_bin"].apply(lambda interval: interval.mid).astype(float)
            summary["x_label"] = summary["_bin"].astype(str)

        else:
            summary = (
                work.groupby(column, dropna=False)[error_col]
                .agg(median_ae="median", n="size")
                .reset_index()
            )

            summary = summary[summary["n"] >= min_count]
            summary["x"] = summary[column]
            summary["x_label"] = summary[column].astype(str)

        summary["df"] = name
        summaries.append(summary)

    summary_df = pd.concat(summaries, ignore_index=True)

    fig, ax = plt.subplots(figsize=figsize)

    for name, group in summary_df.groupby("df"):
        group = group.sort_values("x")

        ax.plot(
            group["x"],
            group["median_ae"],
            marker="o",
            linewidth=2,
            label=name,
        )

    ax.set_xlabel(column)
    ax.set_ylabel("Median absolute error")
    ax.set_title(f"Median absolute error as a function of {column}")
    ax.grid(alpha=0.3)
    ax.legend(title="Dataframe")

    plt.tight_layout()

    return summary_df, ax
