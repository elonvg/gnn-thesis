from typing import Sequence

import torch

from rdkit import Chem
from torch_geometric.data import Data

from rdkit import RDLogger


VIRTUAL_EDGE_FEATURES = (
    "src_is_charged",
    "dst_is_charged",
    "src_is_metal",
    "dst_is_metal",
    "src_is_single_atom_fragment",
    "dst_is_single_atom_fragment",
    "opposite_charge",
    "same_charge",
    "src_formal_charge",
    "dst_formal_charge",
)
VIRTUAL_EDGE_DIM = len(VIRTUAL_EDGE_FEATURES)


def _is_metal_atomic_num(num):
    return (3 <= num <= 4) or (11 <= num <= 13) or (19 <= num <= 31) or \
           (37 <= num <= 50) or (55 <= num <= 84) or (num >= 87)


def _atom_virtual_flags(atom, fragment_size):
    charge = atom.GetFormalCharge()
    is_charged = charge != 0
    is_metal = _is_metal_atomic_num(atom.GetAtomicNum())
    is_single_atom_fragment = fragment_size == 1
    return is_charged, is_metal, is_single_atom_fragment, charge


def _is_virtual_anchor(atom, fragment_size):
    is_charged, is_metal, is_single_atom_fragment, _ = _atom_virtual_flags(atom, fragment_size)
    return is_charged or is_metal or is_single_atom_fragment


def _virtual_edge_attr(mol, fragments_by_atom, src_idx, dst_idx):
    src_atom = mol.GetAtomWithIdx(src_idx)
    dst_atom = mol.GetAtomWithIdx(dst_idx)
    src_fragment_size = fragments_by_atom[src_idx]
    dst_fragment_size = fragments_by_atom[dst_idx]

    src_is_charged, src_is_metal, src_is_single_atom_fragment, src_charge = _atom_virtual_flags(
        src_atom,
        src_fragment_size,
    )
    dst_is_charged, dst_is_metal, dst_is_single_atom_fragment, dst_charge = _atom_virtual_flags(
        dst_atom,
        dst_fragment_size,
    )

    opposite_charge = src_charge * dst_charge < 0
    same_charge = src_charge != 0 and src_charge == dst_charge

    return [
        float(src_is_charged),
        float(dst_is_charged),
        float(src_is_metal),
        float(dst_is_metal),
        float(src_is_single_atom_fragment),
        float(dst_is_single_atom_fragment),
        float(opposite_charge),
        float(same_charge),
        float(src_charge),
        float(dst_charge),
    ]


def _build_virtual_edges(mol, fragments, max_context_atoms=4):
    """Build sparse cross-fragment context edges for charged/metal/single atoms."""
    if len(fragments) <= 1:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, VIRTUAL_EDGE_DIM), dtype=torch.float),
        )

    fragment_infos = []
    fragments_by_atom = {}

    for fragment in fragments:
        atoms = list(fragment)
        fragment_size = len(atoms)
        anchors = [
            atom_idx
            for atom_idx in atoms
            if _is_virtual_anchor(mol.GetAtomWithIdx(atom_idx), fragment_size)
        ]

        for atom_idx in atoms:
            fragments_by_atom[atom_idx] = fragment_size

        fragment_infos.append({
            "atoms": atoms,
            "anchors": anchors,
        })

    virtual_edges = []
    virtual_attrs = []

    for src_frag_idx, src_info in enumerate(fragment_infos):
        for dst_frag_idx, dst_info in enumerate(fragment_infos):
            if src_frag_idx == dst_frag_idx:
                continue

            src_has_anchors = len(src_info["anchors"]) > 0
            dst_has_anchors = len(dst_info["anchors"]) > 0
            if not src_has_anchors and not dst_has_anchors:
                continue

            src_nodes = src_info["anchors"] if src_has_anchors else src_info["atoms"][:max_context_atoms]
            dst_nodes = dst_info["anchors"] if dst_has_anchors else dst_info["atoms"][:max_context_atoms]

            for src_idx in src_nodes:
                for dst_idx in dst_nodes:
                    virtual_edges.append([src_idx, dst_idx])
                    virtual_attrs.append(
                        _virtual_edge_attr(mol, fragments_by_atom, src_idx, dst_idx)
                    )

    if not virtual_edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, VIRTUAL_EDGE_DIM), dtype=torch.float),
        )

    return (
        torch.tensor(virtual_edges, dtype=torch.long).t().contiguous(),
        torch.tensor(virtual_attrs, dtype=torch.float),
    )


def simple_featurizer(
    smiles: str,
    atom_features: Sequence[str] = (
        "atomic_num_scaled",
        "degree",
        "formal_charge",
        "num_hs",
        "is_aromatic",
        "is_in_ring",
        "mass_scaled",
    ),
    bond_features: Sequence[str] = ("bond_order", "is_conjugated", "is_in_ring"),
    add_virtual_edges: bool = True,
    max_virtual_context_atoms: int = 4,
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

    fragments = Chem.GetMolFrags(mol, asMols=False)
    fragment_id = torch.empty(mol.GetNumAtoms(), dtype=torch.long)
    for frag_idx, atom_indices in enumerate(fragments):
        fragment_id[list(atom_indices)] = frag_idx

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

    if add_virtual_edges:
        virtual_edge_index, virtual_edge_attr = _build_virtual_edges(
            mol,
            fragments,
            max_context_atoms=max_virtual_context_atoms,
        )
    else:
        virtual_edge_index = torch.empty((2, 0), dtype=torch.long)
        virtual_edge_attr = torch.empty((0, VIRTUAL_EDGE_DIM), dtype=torch.float)

    features = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        virtual_edge_index=virtual_edge_index,
        virtual_edge_attr=virtual_edge_attr,
        fragment_id=fragment_id,
        num_fragments=torch.tensor([len(fragments)], dtype=torch.long),
        smiles=smiles,
    )

    return features


def simple_featurizer_onehot(
    smiles: str,
    atom_features: Sequence[str] = (
        "atomic_num_scaled",
        "degree",
        "formal_charge",
        "num_hs",
        "is_aromatic",
        "is_in_ring",
    ),
    bond_features: Sequence[str] = ("bond_order", "is_conjugated", "is_in_ring"),
    add_virtual_edges: bool = True,
    max_virtual_context_atoms: int = 4,
    ):
    """Build a PyG graph with one-hot encoded atom and bond features.

    The API and returned fields mirror simple_featurizer. Scaled numeric
    aliases are encoded with the same underlying categorical value.
    """

    RDLogger.DisableLog("rdApp.*")

    def one_hot(value, choices: Sequence, include_unknown: bool = True) -> list[float]:
        encoded = [float(value == choice) for choice in choices]
        if include_unknown:
            encoded.append(float(value not in choices))
        return encoded

    ATOM_FEATURES = {
        "atomic_num":             (lambda a: a.GetAtomicNum(), tuple(range(1, 119)), True),
        "atomic_num_scaled":      (lambda a: a.GetAtomicNum(), tuple(range(1, 119)), True),
        "degree":                 (lambda a: a.GetTotalDegree(), tuple(range(0, 7)), True),
        "formal_charge":          (lambda a: a.GetFormalCharge(), tuple(range(-5, 6)), True),
        "num_hs":                 (lambda a: a.GetTotalNumHs(), tuple(range(0, 5)), True),
        "num_radical_electrons":  (lambda a: a.GetNumRadicalElectrons(), tuple(range(0, 5)), True),
        "is_aromatic":            (lambda a: a.GetIsAromatic(), (False, True), False),
        "is_in_ring":             (lambda a: a.IsInRing(), (False, True), False),
        "hybridization":          (
            lambda a: a.GetHybridization(),
            (
                Chem.HybridizationType.SP,
                Chem.HybridizationType.SP2,
                Chem.HybridizationType.SP3,
                Chem.HybridizationType.SP3D,
                Chem.HybridizationType.SP3D2,
            ),
            True,
        ),
        "chiral_tag":             (
            lambda a: a.GetChiralTag(),
            (
                Chem.ChiralType.CHI_UNSPECIFIED,
                Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.ChiralType.CHI_OTHER,
            ),
            True,
        ),
    }

    BOND_FEATURES = {
        "bond_order":    (lambda b: b.GetBondTypeAsDouble(), (1.0, 1.5, 2.0, 3.0), True),
        "bond_type":     (
            lambda b: b.GetBondType(),
            (
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            ),
            True,
        ),
        "is_conjugated": (lambda b: b.GetIsConjugated(), (False, True), False),
        "is_in_ring":    (lambda b: b.IsInRing(), (False, True), False),
        "is_aromatic":   (lambda b: b.GetIsAromatic(), (False, True), False),
        "stereo":        (
            lambda b: b.GetStereo(),
            (
                Chem.BondStereo.STEREONONE,
                Chem.BondStereo.STEREOANY,
                Chem.BondStereo.STEREOZ,
                Chem.BondStereo.STEREOE,
            ),
            True,
        ),
    }

    unknown_atom = [f for f in atom_features if f not in ATOM_FEATURES]
    unknown_bond = [f for f in bond_features if f not in BOND_FEATURES]
    if unknown_atom or unknown_bond:
        raise KeyError(f"Unknown atom features: {unknown_atom}, bond features: {unknown_bond}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    fragments = Chem.GetMolFrags(mol, asMols=False)
    fragment_id = torch.empty(mol.GetNumAtoms(), dtype=torch.long)
    for frag_idx, atom_indices in enumerate(fragments):
        fragment_id[list(atom_indices)] = frag_idx

    atom_feature_dim = sum(
        len(ATOM_FEATURES[f][1]) + int(ATOM_FEATURES[f][2])
        for f in atom_features
    )
    bond_feature_dim = sum(
        len(BOND_FEATURES[f][1]) + int(BOND_FEATURES[f][2])
        for f in bond_features
    )

    atom_rows = []
    for atom in mol.GetAtoms():
        row = []
        for feature_name in atom_features:
            getter, choices, include_unknown = ATOM_FEATURES[feature_name]
            row.extend(one_hot(getter(atom), choices, include_unknown))
        atom_rows.append(row)

    if atom_rows:
        x = torch.tensor(atom_rows, dtype=torch.float)
    else:
        x = torch.empty((0, atom_feature_dim), dtype=torch.float)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = []
        for feature_name in bond_features:
            getter, choices, include_unknown = BOND_FEATURES[feature_name]
            feat.extend(one_hot(getter(bond), choices, include_unknown))
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_feature_dim), dtype=torch.float)

    if add_virtual_edges:
        virtual_edge_index, virtual_edge_attr = _build_virtual_edges(
            mol,
            fragments,
            max_context_atoms=max_virtual_context_atoms,
        )
    else:
        virtual_edge_index = torch.empty((2, 0), dtype=torch.long)
        virtual_edge_attr = torch.empty((0, VIRTUAL_EDGE_DIM), dtype=torch.float)

    features = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        virtual_edge_index=virtual_edge_index,
        virtual_edge_attr=virtual_edge_attr,
        fragment_id=fragment_id,
        num_fragments=torch.tensor([len(fragments)], dtype=torch.long),
        smiles=smiles,
    )

    return features
