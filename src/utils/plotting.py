import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import seaborn as sns



def plot_smiles(df, figsize=[12, 6]):
    # Count occurrences of each unique SMILES string
    counts = Counter(df['SMILES'])
    plot_df = pd.DataFrame(counts.items(), columns=['SMILES', 'Count'])

    plot_df = plot_df.sort_values('Count', ascending=False)

    # Plotting
    plt.figure(figsize=figsize)
    sns.barplot(data=plot_df, x='SMILES', y='Count', palette='viridis')

    plt.title('Number of Occurrences of Each Molecule')
    plt.xlabel('SMILES String')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def plot_metals(df):

    # List of common metals
    metals = [
        'Ag','Al', 'As', 'Ba', 'Ca', 'Cd', 'Co', 'Cr', 'Cu', 'Fe', 'Hg', 
        'K', 'Li', 'Mg', 'Mn', 'Na', 'Ni', 'Pb', 'Pt', 'Sn', 'Zn', 'Sb'
    ]
    
    # Count occurrences for each metal
    metal_counts = {}
    for m in metals:
        count = df['SMILES'].str.contains(m, case=True).sum()
        if count > 0: # Only include if the metal actually exists in your data
            metal_counts[m] = count

    # Convert to Series for easy plotting and sort by value
    counts_series = pd.Series(metal_counts).sort_values(ascending=False)

    if counts_series.empty:
        print("No metals from the list were found in the dataset.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    counts_series.plot(kind='bar', color=sns.color_palette('viridis', len(counts_series)))
    
    plt.title('Distribution of Metals', fontsize=14)
    plt.xlabel('Metal', fontsize=12)
    plt.ylabel('Number of Molecules', fontsize=12)
    plt.xticks(rotation=0) # Keeps element symbols upright and easy to read
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('metal_distribution.png') # Saves the plot in your project folder
    plt.show()

def plot_toxicity_distribution(train_dataset, test_dataset):
    train_y = [g.y.item() for g in train_dataset]
    test_y  = [g.y.item() for g in test_dataset]

    plt.figure(figsize=(8, 4))
    plt.hist(train_y, bins=50, alpha=0.5, label='Train', density=True)
    plt.hist(test_y,  bins=50, alpha=0.5, label='Test', density=True)
    plt.xlabel("log10c")
    plt.legend()
    plt.title("Target distribution: train vs test")
    plt.show()

    print(f"Train mean: {np.mean(train_y):.2f}, std: {np.std(train_y):.2f}")
    print(f"Test mean:  {np.mean(test_y):.2f},  std: {np.std(test_y):.2f}")

def plot_training(history, figsize=[10, 6]):
    plt.figure(figsize=figsize)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_top_categories(series, title, top_n=10, figsize=(10, 5), color_palette='viridis'):
    counts = series.fillna("Missing").astype(str).value_counts().head(top_n).sort_values()

    if counts.empty:
        print(f"No values available for '{title}'.")
        return

    plt.figure(figsize=figsize)
    counts.plot(kind='barh', color=sns.color_palette(color_palette, len(counts)))
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
    missing.plot(kind='barh', color=sns.color_palette('crest', len(missing)))
    plt.title(title)
    plt.xlabel("Missing fraction")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_log_concentration_by_unit(df, unit_col='conc_unit', value_col='conc', top_n=8, figsize=(12, 5)):
    plot_df = df[[unit_col, value_col]].dropna().copy()
    plot_df = plot_df[plot_df[value_col] > 0]

    if plot_df.empty:
        print("No positive concentration values available for plotting.")
        return

    plot_df[unit_col] = plot_df[unit_col].astype(str)
    top_units = plot_df[unit_col].value_counts().head(top_n).index.tolist()
    plot_df = plot_df[plot_df[unit_col].isin(top_units)].copy()
    plot_df['log10_conc'] = np.log10(plot_df[value_col])

    plt.figure(figsize=figsize)
    sns.boxplot(data=plot_df, x=unit_col, y='log10_conc', order=top_units, color='#6aa6a6')
    plt.title("log10(Concentration) by Unit")
    plt.xlabel("Concentration unit")
    plt.ylabel("log10(conc)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
