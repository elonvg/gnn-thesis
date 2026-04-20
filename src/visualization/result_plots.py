import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


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


def _log_wandb_figure(run, key, fig):
    if run is None or wandb is None:
        return

    run.log({key: wandb.Image(fig)})


def plot_group_mae(summary, category, top_n=10, run=None):
    title = f"{category}: model vs train subgroup mean"

    if summary.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No groups to plot", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        _log_wandb_figure(run, f"categories/{category}/model_vs_baseline_mae", fig)
        plt.show()
        return fig

    plot_df = summary.head(top_n).iloc[::-1]
    labels = [f"{group} (n_test={n})" for group, n in zip(plot_df["group"], plot_df["n"])]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, plot_df["baseline_mae"], color="#c7ced6", label="Train subgroup mean")
    ax.barh(labels, plot_df["model_mae"], color="#2f6db3", label="Model")
    ax.set_xlabel("MAE")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    _log_wandb_figure(run, f"categories/{category}/model_vs_baseline_mae", fig)
    plt.show()
    return fig
