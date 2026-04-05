import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem.SaltRemover import SaltRemover
except ImportError:
    Chem = None
    SaltRemover = None


remover = SaltRemover() if SaltRemover is not None else None


def has_metal(smiles):
    if Chem is None:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False

    for atom in mol.GetAtoms():
        num = atom.GetAtomicNum()
        if (3 <= num <= 4) or (11 <= num <= 13) or (19 <= num <= 31) or \
           (37 <= num <= 50) or (55 <= num <= 84) or (num >= 87):
            return True

    return False


def print_mol_types(df):
    n_mols = len(df)
    smiles = df["SMILES"].fillna("")
    n_unique_mols = smiles.nunique()
    n_salts = smiles.str.contains(r"\.").sum()
    n_single_atoms = ((~smiles.str.contains(r"\.")) & smiles.str.fullmatch(r"[A-Z][a-z]?")).sum()
    n_metals = smiles.apply(has_metal).sum() if Chem is not None else None

    print(f"Total molecules: {n_mols}")
    print(f"Unique molecules: {n_unique_mols}")
    print(f"Salts: {n_salts}, {n_salts / n_mols:.2%}")
    print(f"Single atoms: {n_single_atoms}, {n_single_atoms / n_mols:.2%}")
    if n_metals is None:
        print("Metals: unavailable (RDKit not installed in this environment)")
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


def preprocess(df, split_salts=False, remove_lone=False, remove_metals=False):
    if split_salts:
        df["SMILES"] = df["SMILES"].apply(salt_remover)

    if remove_lone:
        is_lone_atom = (~df["SMILES"].str.contains(r"\.")) & (df["SMILES"].str.fullmatch(r"[A-Z][a-z]?"))
        df = df[~is_lone_atom].reset_index(drop=True)

    if remove_metals:
        df = df[~df["SMILES"].apply(has_metal)].reset_index(drop=True)

    df = preprocess_conc(df)

    return df


def preprocess_conc(df):
    df["log10c"] = df["conc"].apply(lambda x: np.log10(x))
    return df
