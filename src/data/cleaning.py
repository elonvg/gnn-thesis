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
    df = df.copy()

    # Molecule cleanup
    df = preprocess_smiles(df, split_salts=split_salts)

    # Mask data
    mask = mask_data(
        df,
        remove_lone=remove_lone,
        remove_metals=remove_metals,
        max_conc_value=max_conc_value,
        max_duration_hours=max_duration_hours,
        print_summary=False,
    )
    df = df.loc[mask].reset_index(drop=True)

    # Numeric target and duration transforms
    df = preprocess_conc(df)

    df = preprocess_duration(
        df,
        fill_value=duration_fill_value,
        log_transform=log_transform_duration,
        keep_raw=keep_duration_raw,
    )

    return df


def preprocess_smiles(df, split_salts=False):
    if split_salts:
        df["SMILES"] = df["SMILES"].apply(salt_remover)
    return df


def preprocess_conc(df):
    if "conc" not in df.columns:
        return df

    conc = pd.to_numeric(df["conc"], errors="coerce")

    df["conc"] = conc
    df["log10c"] = np.log10(df["conc"])
    return df


def preprocess_duration(
    df,
    fill_value=None,
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
    require_taxid=False,
    remove_lone=False,
    remove_metals=False,
    max_conc_value=None,
    max_duration_hours=None,
    print_summary=True,
):
    filters = filters or {}
    mask = df["SMILES"].notna()

    if print_summary:
        print("Filters")

    def add_mask(label, step_mask):
        nonlocal mask
        mask &= step_mask
        if print_summary:
            vc = step_mask.value_counts(normalize=True)
            print(f"{label}\nTrue: {vc.get(True, 0):.3f}")

    if "conc" in df.columns:
        conc = pd.to_numeric(df["conc"], errors="coerce")
        conc_mask = conc.gt(0)
        if max_conc_value is not None:
            conc_mask &= conc.le(max_conc_value)

        label = "conc > 0"
        if max_conc_value is not None:
            label = f"{label} and <= {max_conc_value}"
        add_mask(label, conc_mask)

    for col, values in filters.items():
        if col in df.columns:
            col_mask = df[col].isin(values)
            add_mask(f"{col}: {values}", col_mask)

    if require_duration:
        if "duration" not in df.columns:
            raise KeyError("'duration' column is required when require_duration=True")
        duration_mask = df["duration"].notna()
        add_mask(f"require_duration: {require_duration}", duration_mask)

    if require_taxid:
        if "taxid" not in df.columns:
            raise KeyError("'taxid' column is required when require_taxid=True")
        taxid_mask = df["taxid"].notna()
        add_mask(f"require_taxid: {require_taxid}", taxid_mask)

    if max_duration_hours is not None and "duration" in df.columns:
        duration = pd.to_numeric(df["duration"], errors="coerce")
        duration_for_filter = duration.where(duration > 0)
        duration_mask = duration_for_filter.isna() | duration_for_filter.le(max_duration_hours)
        add_mask(f"duration <= {max_duration_hours} h or missing", duration_mask)

    if remove_lone:
        non_single_node_mask = ~df["SMILES"].apply(is_single_node)
        add_mask(f"remove_lone: {remove_lone}", non_single_node_mask)

    if remove_metals:
        non_metal_mask = ~df["SMILES"].apply(has_metal)
        add_mask(f"remove_metals: {remove_metals}", non_metal_mask)

    return mask


def rename_columns(df, rename_map=None, int_cols=None):
    df = df.copy()

    if rename_map is None:
        rename_map = {
            "species_group_corrected": "species_group",
            "organism_lifestage_categorized": "organism_lifestage",
            "administration_route_categorized": "administration_route",
            "NCBI_rank_superkingdom": "superkingdom",
            "NCBI_rank_kingdom": "kingdom",
            "NCBI_rank_phylum": "phylum",
            "NCBI_rank_subphylum": "subphylum",
            "NCBI_rank_class": "class",
            "NCBI_rank_order": "order",
            "NCBI_rank_family": "family",
            "NCBI_rank_genus": "genus",
            "NCBI_rank_species": "species",
            "NCBI_sci_name": "species_sci_name",
            "NCBI_last_known_rank": "taxid",
        }
    df = df.rename(columns=rename_map)

    if int_cols is None:
        int_cols = ["taxid"]

    existing_int_cols = [col for col in int_cols if col in df.columns]

    for col in existing_int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def sample_rows(df, n_samples, random_state):
    if n_samples is not None and len(df) > n_samples:
        return df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
    return df.reset_index(drop=True)


def process_data(
    df,
    n_samples,
    random_state,
    filters,
    require_duration,
    require_taxid,
    split_salts,
    remove_lone,
    remove_metals,
    max_conc_value,
    duration_fill_value,
    max_duration_hours,
    log_transform_duration,
    keep_duration_raw,
):
    df = df.copy()

    # Rename columns and fill missing values
    df = rename_columns(df, int_cols=["taxid"])
    df["organism_lifestage"] = df["organism_lifestage"].fillna("adult")
    df["administration_route"] = df["administration_route"].fillna("fill")
    df["duration_unit"] = df["duration_unit"].fillna("h")

    # Normalize SMILES
    df = preprocess_smiles(df, split_salts=split_salts)

    # Mask data based on filters and criteria, then reset index for the remaining rows.
    mask = mask_data(
        df,
        filters=filters,
        require_duration=require_duration,
        require_taxid=require_taxid,
        remove_lone=remove_lone,
        remove_metals=remove_metals,
        max_conc_value=max_conc_value,
        max_duration_hours=max_duration_hours,
    )
    df_masked = df.loc[mask].reset_index(drop=True)

    print()
    print("Loaded and masked training data")
    print(f"Rows in full data: {len(df):,}")
    print(f"Rows after mask: {len(df_masked):,}")

    # Process remaining data
    df_processed = preprocess(
        df_masked.copy(),
        split_salts=False,
        duration_fill_value=duration_fill_value,
        log_transform_duration=log_transform_duration,
        keep_duration_raw=keep_duration_raw,
    )

    print(f"Rows before preprocessing: {len(df_masked):,}")
    print(f"Rows after preprocessing:  {len(df_processed):,}")
    print(f"Rows removed: {len(df_masked) - len(df_processed):,}")

    # Add some metadata cols
    df_processed["fragment_count"] = df_processed["SMILES"].apply(fragment_count).astype(float)
    df_processed["is_salt"] = df_processed["SMILES"].apply(is_salt).astype(float)
    df_processed["has_metal"] = df_processed["SMILES"].apply(has_metal).astype(float)
    df_processed["is_single_node"] = df_processed["SMILES"].apply(is_single_node).astype(float)
    
    # Sample subset for quicker experiments
    if n_samples is not None and len(df_processed) > n_samples:
        df_processed = df_processed.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
    else:
        df_processed = df_processed.reset_index(drop=True)
    
    return df_processed
