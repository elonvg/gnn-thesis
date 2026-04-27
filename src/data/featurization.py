from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from rdkit import Chem
from torch_geometric.data import Data

from rdkit import RDLogger



def simple_featurizer(
    smiles: str,
    atom_features: list[str] = ("atomic_num_scaled", "degree", "formal_charge", "num_hs", "is_aromatic", "is_in_ring", "mass_scaled"),
    bond_features: list[str] = ("bond_order", "is_conjugated", "is_in_ring"),
    ):

    RDLogger.DisableLog("rdApp.*")

    ATOM_FEATURES = {
        "atomic_num":        lambda a: float(a.GetAtomicNum()),
        "atomic_num_scaled": lambda a: float(a.GetAtomicNum()) / 100.0,
        "mass":              lambda a: float(a.GetMass()),
        "mass_scaled":       lambda a: float(a.GetMass()) / 200.0,
        "degree":            lambda a: float(a.GetTotalDegree()),
        "formal_charge":     lambda a: float(a.GetFormalCharge()),
        "num_hs":            lambda a: float(a.GetTotalNumHs()),
        "num_radical_electrons": lambda a: float(a.GetNumRadicalElectrons()),
        "is_aromatic":       lambda a: float(a.GetIsAromatic()),
        "is_in_ring":        lambda a: float(a.IsInRing()),
    }

    BOND_FEATURES = {
        "bond_order":    lambda b: float(b.GetBondTypeAsDouble()),
        "is_conjugated": lambda b: float(b.GetIsConjugated()),
        "is_in_ring":    lambda b: float(b.IsInRing()),
        "is_aromatic":   lambda b: float(b.GetIsAromatic()),
    }

    unknown_atom = [f for f in atom_features if f not in ATOM_FEATURES]
    unknown_bond = [f for f in bond_features if f not in BOND_FEATURES]
    if unknown_atom or unknown_bond:
        raise KeyError(f"Unknown atom features: {unknown_atom}, bond features: {unknown_bond}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    x = torch.tensor([
        [ATOM_FEATURES[f](atom) for f in atom_features]
        for atom in mol.GetAtoms()
    ], dtype=torch.float)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [BOND_FEATURES[f](bond) for f in bond_features]
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(bond_features)), dtype=torch.float)

    features = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

    return features