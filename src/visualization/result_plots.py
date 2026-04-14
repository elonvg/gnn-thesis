import matplotlib.pyplot as plt
import numpy as np


def summarize_by_group(results_df, train_df, group_col, min_count=20):
    df = results_df.copy()
    df["group"] = df[group_col].astype(str).replace("nan", "Missing")

    train_groups = train_df.copy()
    train_groups["group"] = train_groups[group_col].astype(str).replace("nan", "Missing")

    baseline = train_groups.groupby("group")["log10c"].mean()
    global_mean = train_groups["log10c"].mean()

    df["baseline_log10c"] = df["group"].map(baseline).fillna(global_mean)
    df["model_abs_error"] = (df["pred_log10c"] - df["actual_log10c"]).abs()
    df["baseline_abs_error"] = (df["baseline_log10c"] - df["actual_log10c"]).abs()
    df["model_sq_error"] = (df["pred_log10c"] - df["actual_log10c"]) ** 2
    df["baseline_sq_error"] = (df["baseline_log10c"] - df["actual_log10c"]) ** 2

    summary = (
        df.groupby("group")
        .agg(
            n=("group", "size"),
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


def plot_group_mae(summary, title, top_n=10):
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
