from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def plot_smiles(df, figsize=(12, 6)):
    counts = Counter(df["SMILES"])
    plot_df = pd.DataFrame(counts.items(), columns=["SMILES", "Count"]).sort_values("Count", ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(data=plot_df, x="SMILES", y="Count", palette="viridis")
    plt.title("Number of Occurrences of Each Molecule")
    plt.xlabel("SMILES String")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_metals(df, save_path=None):
    metals = [
        "Ag", "Al", "As", "Ba", "Ca", "Cd", "Co", "Cr", "Cu", "Fe", "Hg",
        "K", "Li", "Mg", "Mn", "Na", "Ni", "Pb", "Pt", "Sn", "Zn", "Sb",
    ]

    metal_counts = {}
    for metal in metals:
        count = df["SMILES"].str.contains(metal, case=True).sum()
        if count > 0:
            metal_counts[metal] = count

    counts_series = pd.Series(metal_counts).sort_values(ascending=False)

    if counts_series.empty:
        print("No metals from the list were found in the dataset.")
        return

    plt.figure(figsize=(10, 6))
    counts_series.plot(kind="bar", color=sns.color_palette("viridis", len(counts_series)))
    plt.title("Distribution of Metals", fontsize=14)
    plt.xlabel("Metal", fontsize=12)
    plt.ylabel("Number of Molecules", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path is None:
        save_path = DEFAULT_FIGURES_DIR / "metal_distribution.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.show()


def plot_top_categories(series, title, top_n=10, figsize=(10, 5), color_palette="viridis"):
    counts = series.fillna("Missing").astype(str).value_counts().head(top_n).sort_values()

    if counts.empty:
        print(f"No values available for '{title}'.")
        return

    plt.figure(figsize=figsize)
    counts.plot(kind="barh", color=sns.color_palette(color_palette, len(counts)))
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_missing_fraction(df, columns=None, title="Missing Fraction by Column", figsize=(10, 5)):
    if columns is None:
        columns = df.columns

    missing = df[columns].isna().mean().sort_values()

    plt.figure(figsize=figsize)
    missing.plot(kind="barh", color=sns.color_palette("crest", len(missing)))
    plt.title(title)
    plt.xlabel("Missing fraction")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_fractions(series, title, top_n=None, figsize=(9, 4), color="#4c78a8"):
    fractions = series.fillna("Missing").astype(str).value_counts(normalize=True)
    if top_n is not None:
        fractions = fractions.head(top_n)
    summary = fractions.rename("fraction").to_frame()
    ax = summary["fraction"].sort_values().plot(kind="barh", figsize=figsize, color=color)
    ax.set_title(title)
    ax.set_xlabel("Fraction of rows")
    ax.set_ylabel("")
    ax.set_xlim(0, max(0.01, summary["fraction"].max() * 1.1))
    ax.figure.tight_layout()
    
    return summary


def plot_log_concentration_by_unit(df, unit_col="conc_unit", value_col="conc", top_n=8, figsize=(12, 5)):
    plot_df = df[[unit_col, value_col]].dropna().copy()
    plot_df = plot_df[plot_df[value_col] > 0]

    if plot_df.empty:
        print("No positive concentration values available for plotting.")
        return

    plot_df[unit_col] = plot_df[unit_col].astype(str)
    top_units = plot_df[unit_col].value_counts().head(top_n).index.tolist()
    plot_df = plot_df[plot_df[unit_col].isin(top_units)].copy()
    plot_df["log10_conc"] = np.log10(plot_df[value_col])

    plt.figure(figsize=figsize)
    sns.boxplot(data=plot_df, x=unit_col, y="log10_conc", order=top_units, color="#6aa6a6")
    plt.title("log10(Concentration) by Unit")
    plt.xlabel("Concentration unit")
    plt.ylabel("log10(conc)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
