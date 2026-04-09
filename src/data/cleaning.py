import re
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.SaltRemover import SaltRemover
except ImportError:
    Chem = None
    SaltRemover = None


remover = SaltRemover() if SaltRemover is not None else None

_SINGLE_ATOM_PATTERN = re.compile(r"[A-Z][a-z]?")
_SINGLE_BRACKET_PATTERN = re.compile(r"\[[^\[\]\.]+\]")


def _is_metal_atomic_num(num):
    return (3 <= num <= 4) or (11 <= num <= 13) or (19 <= num <= 31) or \
           (37 <= num <= 50) or (55 <= num <= 84) or (num >= 87)


def _fallback_single_node(smiles):
    if not isinstance(smiles, str) or not smiles or "." in smiles:
        return False
    return bool(
        _SINGLE_ATOM_PATTERN.fullmatch(smiles)
        or _SINGLE_BRACKET_PATTERN.fullmatch(smiles)
    )


def _fallback_fragment_count(smiles):
    if not isinstance(smiles, str) or not smiles:
        return 0
    return sum(1 for fragment in smiles.split(".") if fragment)


@lru_cache(maxsize=100_000)
def _smiles_stats_cached(smiles):
    if Chem is None:
        fragment_count = _fallback_fragment_count(smiles)
        atom_count = 1 if _fallback_single_node(smiles) else 0
        return fragment_count, atom_count, False

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        fragment_count = _fallback_fragment_count(smiles)
        atom_count = 1 if _fallback_single_node(smiles) else 0
        return fragment_count, atom_count, False

    fragment_count = len(Chem.GetMolFrags(mol))
    atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_count = len(atom_numbers)
    has_metal_flag = any(_is_metal_atomic_num(num) for num in atom_numbers)
    return fragment_count, atom_count, has_metal_flag


def _smiles_stats(smiles):
    if not isinstance(smiles, str) or not smiles:
        return 0, 0, False
    return _smiles_stats_cached(smiles)


def fragment_count(smiles):
    return _smiles_stats(smiles)[0]


def is_salt(smiles):
    return fragment_count(smiles) > 1


def is_single_node(smiles):
    return _smiles_stats(smiles)[1] == 1


def has_metal(smiles):
    return _smiles_stats(smiles)[2]


def print_mol_types(df):
    n_mols = len(df)
    smiles = df["SMILES"].fillna("")
    n_unique_mols = smiles.nunique()
    n_salts = smiles.apply(is_salt).sum()
    n_single_nodes = smiles.apply(is_single_node).sum()
    n_metals = smiles.apply(has_metal).sum()

    print(f"Total molecules: {n_mols}")
    print(f"Unique molecules: {n_unique_mols}")
    print(f"Disconnected species: {n_salts}, {n_salts / n_mols:.2%}")
    print(f"Single-node species: {n_single_nodes}, {n_single_nodes / n_mols:.2%}")
    if Chem is None:
        print("Metals: unavailable without RDKit")
    else:
        print(f"Metals: {n_metals}, {n_metals / n_mols:.2%}")


def keep_largest(smile):
    if Chem is None:
        raise ImportError("rdkit is required for keep_largest() but is not installed in this environment.")

    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(smile), asMols=True)
    largest = None
    largest_size = 0

    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_size:
            largest_size = size
            largest = mol

    return Chem.MolToSmiles(largest) if largest else None


def salt_remover(smile, remover=remover):
    if Chem is None or remover is None:
        raise ImportError("rdkit is required for salt_remover() but is not installed in this environment.")

    smile = Chem.MolToSmiles(
        remover.StripMol(Chem.MolFromSmiles(smile), dontRemoveEverything=True),
        isomericSmiles=True,
    )
    if "." in smile:
        smile = keep_largest(smile)
    return smile


def preprocess(
    df,
    split_salts=False,
    remove_lone=False,
    remove_metals=False,
    max_conc_value=None,
    duration_fill_value=None,
    max_duration_hours=None,
    log_transform_duration=False,
    keep_duration_raw=False,
):
    if split_salts:
        df["SMILES"] = df["SMILES"].apply(salt_remover)

    if remove_lone:
        is_single_node_mask = df["SMILES"].apply(is_single_node)
        df = df[~is_single_node_mask].reset_index(drop=True)

    if remove_metals:
        df = df[~df["SMILES"].apply(has_metal)].reset_index(drop=True)

    df = preprocess_duration(
        df,
        fill_value=duration_fill_value,
        max_hours=max_duration_hours,
        log_transform=log_transform_duration,
        keep_raw=keep_duration_raw,
    )
    df = preprocess_conc(
        df,
        max_conc=max_conc_value
    )

    return df


def preprocess_conc(df, max_conc):

    if "conc" not in df.columns:
        return df

    conc = pd.to_numeric(df["conc"], errors="coerce")

    if max_conc is not None:
        keep_mask = conc.isna() | conc.le(max_conc)
        df = df.loc[keep_mask].reset_index(drop=True)
        conc = conc.loc[keep_mask].reset_index(drop=True)

    df["conc"] = conc
    df["log10c"] = np.log10(df["conc"])
    return df


def preprocess_duration(
    df,
    fill_value=None,
    max_hours=None,
    log_transform=False,
    keep_raw=False,
):
    if "duration" not in df.columns:
        return df

    duration = pd.to_numeric(df["duration"], errors="coerce")

    if keep_raw:
        df["duration_raw"] = duration

    # Non-positive durations are treated as missing before imputation so the
    # later log10 transform always receives positive values.
    duration = duration.where(duration > 0)

    if max_hours is not None:
        keep_mask = duration.isna() | duration.le(max_hours)
        df = df.loc[keep_mask].reset_index(drop=True)
        duration = duration.loc[keep_mask].reset_index(drop=True)

    if fill_value is not None:
        duration = duration.fillna(fill_value)

    if log_transform:
        if duration.le(0).any():
            raise ValueError("Duration values must be positive before log10 transformation.")
        duration = duration.apply(np.log10)

    df["duration"] = duration
    return df


def mask_data(
    df,
    filters=None,
    require_duration=False,
    require_taxonomy=False,
    taxonomy_columns=None,
):
    filters = filters or {}
    mask = (
        df["conc"].gt(0)
        & df["SMILES"].notna()
    )

    print("Filters")
    for col, values in filters.items():
        if col in df.columns:
            col_mask = df[col].isin(values)
            mask &= col_mask
            vc = col_mask.value_counts(normalize=True)
            print(f"{col}: {values}\nTrue: {vc.get(True, 0):.3f}")

    if require_duration:
        if "duration" not in df.columns:
            raise KeyError("'duration' column is required when require_duration=True")
        duration_mask = df["duration"].notna()
        mask &= duration_mask
        vc = duration_mask.value_counts(normalize=True)
        print(f"require_duration: {require_duration}\nTrue: {vc.get(True, 0):.3f}")

    if require_taxonomy:
        missing_taxonomy_columns = [col for col in taxonomy_columns if col not in df.columns]
        if missing_taxonomy_columns:
            raise KeyError(
                "Missing taxonomy columns required for require_taxonomy=True: "
                f"{missing_taxonomy_columns}"
            )
        taxonomy_mask = df[list(taxonomy_columns)].notna().all(axis=1)
        mask &= taxonomy_mask
        vc = taxonomy_mask.value_counts(normalize=True)
        print(f"require_taxonomy: {require_taxonomy}\nTrue: {vc.get(True, 0):.3f}")

    return mask
