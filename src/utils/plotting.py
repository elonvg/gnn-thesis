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


def plot_training(history, figsize=[10, 6]):
    plt.figure(figsize=figsize)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()