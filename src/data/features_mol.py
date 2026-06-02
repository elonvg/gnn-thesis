import numpy as np
import pandas as pd
import re


from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger


_SINGLE_ATOM_PATTERN = re.compile(r"[A-Z][a-z]?")
_SINGLE_BRACKET_PATTERN = re.compile(r"\[[^\[\]\.]+\]")
_HALOGEN_ATOMIC_NUMS = frozenset({9, 17, 35, 53, 85, 117})
_TRANSITION_METAL_ATOMIC_NUMS = frozenset(
    range(21, 31)
) | frozenset(
    range(39, 49)
) | frozenset(
    range(57, 81)
) | frozenset(
    range(89, 113)
)

MOLECULE_CATEGORICAL_METADATA_COLUMNS = [
    "is_salt",
    "has_metal",
    "has_transition_metal",
    "has_halogen",
    "is_single_node",
]

MOLECULE_NUMERICAL_METADATA_COLUMNS = [
    "fragment_count",
    "mol_weight",
    "log10_mol_weight",
    "logp",
    "tpsa",
    "log10_tpsa_plus1",
    "h_bond_donor_count",
    "h_bond_acceptor_count",
    "heavy_atom_count",
    "log10_heavy_atom_count_plus1",
    "hetero_atom_count",
    "halogen_count",
    "metal_count",
    "transition_metal_count",
    "ring_count",
    "aromatic_ring_count",
    "rotatable_bond_count",
    "formal_charge",
]


def _is_metal_atomic_num(num):
    return (3 <= num <= 4) or (11 <= num <= 13) or (19 <= num <= 31) or \
           (37 <= num <= 50) or (55 <= num <= 84) or (num >= 87)


def _is_transition_metal_atomic_num(num):
    return num in _TRANSITION_METAL_ATOMIC_NUMS


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


# @lru_cache(maxsize=100_000)
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

def _safe_log10(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value) or value <= 0:
        return np.nan
    return float(np.log10(value))


def _safe_log10_plus1(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value) or value < 0:
        return np.nan
    return float(np.log10(value + 1.0))


def _descriptor_defaults(fragment_count_value=0, atom_count=0, has_metal_flag=False):
    return {
        "fragment_count": float(fragment_count_value),
        "mol_weight": np.nan,
        "log10_mol_weight": np.nan,
        "logp": np.nan,
        "tpsa": np.nan,
        "log10_tpsa_plus1": np.nan,
        "h_bond_donor_count": np.nan,
        "h_bond_acceptor_count": np.nan,
        "heavy_atom_count": float(atom_count),
        "log10_heavy_atom_count_plus1": _safe_log10_plus1(atom_count),
        "hetero_atom_count": np.nan,
        "halogen_count": 0.0,
        "metal_count": float(has_metal_flag),
        "transition_metal_count": 0.0,
        "ring_count": np.nan,
        "aromatic_ring_count": np.nan,
        "rotatable_bond_count": np.nan,
        "formal_charge": np.nan,
        "is_salt": float(fragment_count_value > 1),
        "has_metal": float(has_metal_flag),
        "has_transition_metal": 0.0,
        "has_halogen": 0.0,
        "is_single_node": float(atom_count == 1),
    }


def _metadata_tuple(metadata, metadata_cols):
    return tuple(metadata[col] for col in metadata_cols)


# @lru_cache(maxsize=100_000)
def _molecule_metadata_cached(smiles, metadata_cols):
    fragment_count_value, atom_count, has_metal_flag = _smiles_stats(smiles)
    metadata = _descriptor_defaults(
        fragment_count_value=fragment_count_value,
        atom_count=atom_count,
        has_metal_flag=has_metal_flag,
    )

    if Chem is None:
        return _metadata_tuple(metadata, metadata_cols)

    mol = Chem.MolFromSmiles(smiles)
    atom_mol = mol if mol is not None else Chem.MolFromSmiles(smiles, sanitize=False)

    if atom_mol is not None:
        atoms = list(atom_mol.GetAtoms())
        atom_numbers = [atom.GetAtomicNum() for atom in atoms]
        heavy_atom_count_value = float(sum(num > 1 for num in atom_numbers))
        metal_count_value = sum(_is_metal_atomic_num(num) for num in atom_numbers)
        transition_metal_count_value = sum(
            _is_transition_metal_atomic_num(num) for num in atom_numbers
        )
        halogen_count_value = sum(num in _HALOGEN_ATOMIC_NUMS for num in atom_numbers)

        metadata.update({
            "heavy_atom_count": heavy_atom_count_value,
            "log10_heavy_atom_count_plus1": _safe_log10_plus1(heavy_atom_count_value),
            "hetero_atom_count": float(sum(num not in (1, 6) for num in atom_numbers)),
            "halogen_count": float(halogen_count_value),
            "metal_count": float(metal_count_value),
            "transition_metal_count": float(transition_metal_count_value),
            "formal_charge": float(sum(atom.GetFormalCharge() for atom in atoms)),
            "has_metal": float(metal_count_value > 0),
            "has_transition_metal": float(transition_metal_count_value > 0),
            "has_halogen": float(halogen_count_value > 0),
        })

    if mol is not None:
        mol_weight_value = float(rdMolDescriptors.CalcExactMolWt(mol))
        tpsa_value = float(rdMolDescriptors.CalcTPSA(mol))

        metadata.update({
            "mol_weight": mol_weight_value,
            "log10_mol_weight": _safe_log10(mol_weight_value),
            "logp": float(rdMolDescriptors.CalcCrippenDescriptors(mol)[0]),
            "tpsa": tpsa_value,
            "log10_tpsa_plus1": _safe_log10_plus1(tpsa_value),
            "h_bond_donor_count": float(rdMolDescriptors.CalcNumHBD(mol)),
            "h_bond_acceptor_count": float(rdMolDescriptors.CalcNumHBA(mol)),
            "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
            "aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "rotatable_bond_count": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        })

    return _metadata_tuple(metadata, metadata_cols)


def molecule_metadata(smiles, metadata_cols):

    values = _molecule_metadata_cached(smiles, metadata_cols)
    return dict(zip(metadata_cols, values))


def add_molecule_metadata(df, categorical_cols=MOLECULE_CATEGORICAL_METADATA_COLUMNS, numerical_cols=MOLECULE_NUMERICAL_METADATA_COLUMNS):

    RDLogger.DisableLog("rdApp.*")

    metadata_cols = (categorical_cols + numerical_cols)

    metadata = pd.DataFrame(
        [molecule_metadata(smiles, metadata_cols) for smiles in df["SMILES"].fillna("")],
        index=df.index,
    )
    metadata = metadata[metadata_cols].astype(float)

    df = df.copy()
    for col in metadata_cols:
        df[col] = metadata[col]
    return df