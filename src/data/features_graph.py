import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from rdkit import Chem
from torch_geometric.data import Data

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

DEFAULT_ATOM_FEATURES = (
    "atomic_num",
    "degree",
    "formal_charge",
    "num_hs",
    "hybridization",
    "is_aromatic",
    "is_in_ring",
    "atomic_mass",
    "period",
    "group",
    "covalent_radius",
    "vdw_radius",
    "is_metal",
    "is_transition_metal",
    "is_alkali_metal",
    "is_alkaline_earth_metal",
    "is_lanthanoid",
    "is_actinoid",
    "is_post_transition_metal",
)

DEFAULT_BOND_FEATURES = (
    "bond_order",
    "is_conjugated",
    "is_in_ring",
    "stereo",
)

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

@dataclass(frozen=True)
class CategoricalFeature:
    getter: Callable
    choices: Sequence
    include_unknown: bool = True


@dataclass(frozen=True)
class NumericFeature:
    getter: Callable
    scale: float = 1.0
    include_missing: bool = False
    missing_value: float = 0.0

PERIODIC_TABLE = Chem.GetPeriodicTable()

def _is_metal_atomic_num(num):
    return (
        _is_alkali_metal_atomic_num(num)
        or _is_alkaline_earth_metal_atomic_num(num)
        or _is_transition_metal_atomic_num(num)
        or _is_lanthanoid_atomic_num(num)
        or _is_actinoid_atomic_num(num)
        or _is_post_transition_metal_atomic_num(num)
    )

def _is_transition_metal_atomic_num(num):
    return (
        (21 <= num <= 30)
        or (39 <= num <= 48)
        or (72 <= num <= 80)
        or (104 <= num <= 112)
    )


def _is_alkali_metal_atomic_num(num):
    return num in {3, 11, 19, 37, 55, 87}


def _is_alkaline_earth_metal_atomic_num(num):
    return num in {4, 12, 20, 38, 56, 88}


def _is_lanthanoid_atomic_num(num):
    return 57 <= num <= 71


def _is_actinoid_atomic_num(num):
    return 89 <= num <= 103


def _is_post_transition_metal_atomic_num(num):
    return num in {13, 31, 49, 50, 81, 82, 83, 84, 113, 114, 115, 116}


def _is_metalloid_atomic_num(num):
    return num in {5, 14, 32, 33, 51, 52}


def _period_from_atomic_num(num):
    if 1 <= num <= 2:
        return 1
    if 3 <= num <= 10:
        return 2
    if 11 <= num <= 18:
        return 3
    if 19 <= num <= 36:
        return 4
    if 37 <= num <= 54:
        return 5
    if 55 <= num <= 86:
        return 6
    if 87 <= num <= 118:
        return 7
    return None


def _group_from_atomic_num(num):
    if num == 1:
        return 1
    if num == 2:
        return 18
    if 3 <= num <= 10:
        return num - 2 if num <= 4 else num + 8
    if 11 <= num <= 18:
        return num - 10 if num <= 12 else num
    if 19 <= num <= 36:
        return num - 18
    if 37 <= num <= 54:
        return num - 36
    if 55 <= num <= 56:
        return num - 54
    if 57 <= num <= 71:
        return 3
    if 72 <= num <= 86:
        return num - 68
    if 87 <= num <= 88:
        return num - 86
    if 89 <= num <= 103:
        return 3
    if 104 <= num <= 118:
        return num - 100
    return None

def _atom_period(atom):
    return _period_from_atomic_num(atom.GetAtomicNum())


def _atom_group(atom):
    return _group_from_atomic_num(atom.GetAtomicNum())


def _atom_covalent_radius(atom):
    return PERIODIC_TABLE.GetRcovalent(atom.GetAtomicNum())


def _atom_vdw_radius(atom):
    return PERIODIC_TABLE.GetRvdw(atom.GetAtomicNum())

ALL_ATOM_FEATURES = {
    "atomic_num": CategoricalFeature(lambda a: a.GetAtomicNum(), tuple(range(1, 119)), True),
    "atomic_num_numeric": NumericFeature(lambda a: a.GetAtomicNum(), scale=118.0),
    "degree": CategoricalFeature(lambda a: a.GetTotalDegree(), tuple(range(0, 7)), True),
    "formal_charge": CategoricalFeature(lambda a: a.GetFormalCharge(), tuple(range(-5, 6)), True),
    "formal_charge_numeric": NumericFeature(lambda a: a.GetFormalCharge(), scale=5.0),
    "num_hs": CategoricalFeature(lambda a: a.GetTotalNumHs(), tuple(range(0, 5)), True),
    "num_radical_electrons": CategoricalFeature(
        lambda a: a.GetNumRadicalElectrons(),
        tuple(range(0, 5)),
        True,
    ),
    "is_aromatic": CategoricalFeature(lambda a: a.GetIsAromatic(), (False, True), False),
    "is_in_ring": CategoricalFeature(lambda a: a.IsInRing(), (False, True), False),
    "hybridization": CategoricalFeature(
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
    "chiral_tag": CategoricalFeature(
        lambda a: a.GetChiralTag(),
        (
            Chem.ChiralType.CHI_UNSPECIFIED,
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.ChiralType.CHI_OTHER,
        ),
        True,
    ),
    "atomic_mass": NumericFeature(lambda a: a.GetMass(), scale=300.0),
    "period": CategoricalFeature(_atom_period, tuple(range(1, 8)), True),
    "period_numeric": NumericFeature(_atom_period, scale=7.0, include_missing=True),
    "group": CategoricalFeature(_atom_group, tuple(range(1, 19)), True),
    "group_numeric": NumericFeature(_atom_group, scale=18.0, include_missing=True),
    "covalent_radius": NumericFeature(_atom_covalent_radius, scale=3.0),
    "vdw_radius": NumericFeature(_atom_vdw_radius, scale=4.0),
    "is_metal": CategoricalFeature(
        lambda a: _is_metal_atomic_num(a.GetAtomicNum()),
        (False, True),
        False,
    ),
    "is_transition_metal": CategoricalFeature(
        lambda a: _is_transition_metal_atomic_num(a.GetAtomicNum()),
        (False, True),
        False,
    ),
    "is_alkali_metal": CategoricalFeature(
        lambda a: _is_alkali_metal_atomic_num(a.GetAtomicNum()),
        (False, True),
        False,
    ),
    "is_alkaline_earth_metal": CategoricalFeature(
        lambda a: _is_alkaline_earth_metal_atomic_num(a.GetAtomicNum()),
        (False, True),
        False,
    ),
    "is_lanthanoid": CategoricalFeature(
        lambda a: _is_lanthanoid_atomic_num(a.GetAtomicNum()),
        (False, True),
        False,
    ),
    "is_actinoid": CategoricalFeature(
        lambda a: _is_actinoid_atomic_num(a.GetAtomicNum()),
        (False, True),
        False,
    ),
    "is_post_transition_metal": CategoricalFeature(
        lambda a: _is_post_transition_metal_atomic_num(a.GetAtomicNum()),
        (False, True),
        False,
    ),
    "is_metalloid": CategoricalFeature(
        lambda a: _is_metalloid_atomic_num(a.GetAtomicNum()),
        (False, True),
        False,
    ),
}


ALL_BOND_FEATURES = {
    "bond_order": CategoricalFeature(
        lambda b: b.GetBondTypeAsDouble(),
        (1.0, 1.5, 2.0, 3.0),
        True,
    ),
    "bond_order_numeric": NumericFeature(lambda b: b.GetBondTypeAsDouble(), scale=3.0),
    "bond_type": CategoricalFeature(
        lambda b: b.GetBondType(),
        (
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        ),
        True,
    ),
    "is_conjugated": CategoricalFeature(lambda b: b.GetIsConjugated(), (False, True), False),
    "is_in_ring": CategoricalFeature(lambda b: b.IsInRing(), (False, True), False),
    "is_aromatic": CategoricalFeature(lambda b: b.GetIsAromatic(), (False, True), False),
    "stereo": CategoricalFeature(
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

def _one_hot(value, choices, include_unknown=True):
    encoding = [1.0 if value == choice else 0.0 for choice in choices]
    if include_unknown:
        encoding.append(float(value not in choices))
    return encoding

def _numeric(value, scale, include_missing, missing_value):
    missing = value is None # bool

    if not missing:
        try:
            value = float(value)
        except (TypeError, ValueError):
            missing = True
    if not missing and math.isnan(value):
        missing = True
    if missing:
        value = missing_value

    encoding = [float(value) / scale]
    if include_missing:
        encoding.append(float(missing))
    return encoding

def _encode_feature(obj, spec):
    value = spec.getter(obj) # Returns value to be encoded

    if isinstance(spec, CategoricalFeature):
        return _one_hot(value, spec.choices, spec.include_unknown)
    if isinstance(spec, NumericFeature):
        return _numeric(value, scale=spec.scale, include_missing=spec.include_missing, missing_value=spec.missing_value)

def _feature_dim(spec) -> int:
    if isinstance(spec, CategoricalFeature):
        return len(spec.choices) + int(spec.include_unknown)
    if isinstance(spec, NumericFeature):
        return 1 + int(spec.include_missing)
    
def _atom_virtual_flags(atom, fragment_size):
    charge = atom.GetFormalCharge()
    is_charged = charge != 0
    is_metal = _is_metal_atomic_num(atom.GetAtomicNum())
    is_single_atom_fragment = fragment_size == 1
    return is_charged, is_metal, is_single_atom_fragment, charge

def _is_virtual_anchor(atom, fragment_size):
    # Anchor prio: charged atom, metals, single atoms
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
    # Build sparse cross-fragment context edges for charged/metal/single atoms
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
    

class GraphFeaturizer(object):

    def __init__(
            self,
            atom_features=DEFAULT_ATOM_FEATURES,
            bond_features=DEFAULT_BOND_FEATURES,
            add_virtual_edges=True,
            max_virtual_neighbors=3,
            ):
        
        self.atom_features = tuple(atom_features)
        self.bond_features = tuple(bond_features)
        self.add_virtual_edges = add_virtual_edges
        self.max_virtual_neighbors = max_virtual_neighbors
        self.virtual_edge_dim = VIRTUAL_EDGE_DIM if add_virtual_edges else 0

        self.atom_feature_dim = sum(
            _feature_dim(ALL_ATOM_FEATURES[feature])
            for feature in self.atom_features
        )
        self.bond_feature_dim = sum(
            _feature_dim(ALL_BOND_FEATURES[feature])
            for feature in self.bond_features
        )

        self.graph_cache = {}

    def featurize(
            self,
            smiles,
    ):
        
        if smiles in self.graph_cache:
            return self.graph_cache[smiles].clone()

        mol = Chem.MolFromSmiles(smiles)

        # Tensor showing what fragment each atom belongs to
        fragments = Chem.GetMolFrags(mol)
        num_fragments = torch.tensor([len(fragments)], dtype=torch.long)
        fragment_id = torch.empty(mol.GetNumAtoms(), dtype=torch.long)
        for frag_idx, atom_indices in enumerate(fragments):
            fragment_id[list(atom_indices)] = frag_idx
        
        
        atom_rows = []
        for atom in mol.GetAtoms():
            row = []
            for feature in self.atom_features:
                row.extend(_encode_feature(atom, ALL_ATOM_FEATURES[feature]))
            atom_rows.append(row)
        
        x = torch.tensor(atom_rows, dtype=torch.float)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat = []
            for feature in self.bond_features:
                feat.extend(_encode_feature(bond, ALL_BOND_FEATURES[feature]))
            edge_indices += [[i, j], [j, i]]
            edge_attrs += [feat, feat]
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self.bond_feature_dim), dtype=torch.float)

        if self.add_virtual_edges:
            virtual_edge_index, virtual_edge_attr = _build_virtual_edges(
                mol,
                fragments,
                self.max_virtual_neighbors
            )
        else:
            virtual_edge_index = torch.empty((2, 0), dtype=torch.long)
            virtual_edge_attr = torch.empty((0, self.virtual_edge_dim), dtype=torch.float)

        graph_features = Data(
            x = x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            virtual_edge_index = virtual_edge_index,
            virtual_edge_attr = virtual_edge_attr,
            fragment_id=fragment_id,
            num_fragments=num_fragments,
            smiles=smiles,
        )

        self.graph_cache[smiles] = graph_features

        return self.graph_cache[smiles].clone()
    
    def get_graph_cache(self):
        return self.graph_cache


if __name__ == "__main__":
    smiles = "C#CC(C)=O.[O]"

    featurizer = GraphFeaturizer()

    graph_features = featurizer.featurize(smiles)

    print(graph_features)