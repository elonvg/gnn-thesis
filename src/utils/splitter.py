import torch
import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

def generate_scaffold(smiles, include_chirality=False):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold


def scaffold_split(features, frac_train=0.8, frac_test=0.2, frac_valid=0.0, seed=None):
    
    assert abs(frac_train + frac_test + frac_valid - 1.0) < 1e-6
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Generate scaffolds for all molecules
    # create dict of the form {scaffold_i: [idx1, idx....]}
    n = len(features)
    all_scaffolds = {}
    for i in range(n):
        scaffold = generate_scaffold(smiles=features[i].smiles, include_chirality=False)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # Sort scaffold from largest to smallest sets
    scaffold_sets = sorted(all_scaffolds.values(), key=lambda x: len(x), reverse=True)

    train_indices, test_indices, val_indices = [], [], []
    train_cutoff = frac_train * n
    val_cutoff = (frac_train + frac_valid) * n

    # Assign scaffolds to train, validation, and test sets
    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= train_cutoff:
            train_indices.extend(scaffold_set)
        elif len(train_indices) + len(val_indices) + len(scaffold_set) <= val_cutoff:
            val_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    # Map indices to datasets
    train_dataset = [features[i] for i in train_indices]
    test_dataset = [features[i] for i in test_indices]
    val_dataset = [features[i] for i in val_indices]

    return train_dataset, test_dataset, val_dataset

    