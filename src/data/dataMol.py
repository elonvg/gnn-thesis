import pandas as pd
import numpy as np
try:
    from rdkit import Chem
except ImportError:
    Chem = None

try:
    import deepchem as dc
except ImportError:
    dc = None

def load_data(path, selected_columns=None, cut=None):
    # Load data
    df = pd.read_csv(path, low_memory=False)

    # Select columns
    if selected_columns is None:
        selected_columns = ["SMILES", "conc"]
    
    df = df[selected_columns]

    # Slice data
    if cut is not None: 
        # This applies the slice [0:cut]
        df = df.iloc[:cut]

    return df

def has_metal(smiles):
    if Chem is None:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if not mol: return False
    
    for atom in mol.GetAtoms():
        # Get the atomic number
        num = atom.GetAtomicNum()
        # Most metals have atomic numbers in these ranges:
        # 3-4, 11-13, 19-31, 37-50, 55-84, 87+
        if (3 <= num <= 4) or (11 <= num <= 13) or (19 <= num <= 31) or \
           (37 <= num <= 50) or (55 <= num <= 84) or (num >= 87):
            return True
    return False

def print_mol_types(df):
    n_mols = len(df)
    n_unique_mols = df['SMILES'].nunique()
    n_salts = df['SMILES'].str.contains(r'\.').sum()
    n_single_atoms = (
        (~df['SMILES'].str.contains(r'\.')) & 
        (df['SMILES'].str.fullmatch(r'[A-Z][a-z]?'))
    ).sum()
    n_metals = df['SMILES'].apply(has_metal).sum() if Chem is not None else None

    print(f"Total molecules: {n_mols}")
    print(f"Unique molecules: {n_unique_mols}")
    print(f"Salts: {n_salts}, {n_salts/n_mols:.2%}")
    print(f"Single atoms: {n_single_atoms}, {n_single_atoms/n_mols:.2%}")
    if n_metals is None:
        print("Metals: unavailable (RDKit not installed in this environment)")
    else:
        print(f"Metals: {n_metals}, {n_metals/n_mols:.2%}")

def load_base_dataframe(config):
    return load_data(config["path"], config["selected_columns"], config["cut"])


def featurize(df, featurizer, apply_filter=False):
    if dc is None:
        raise ImportError("deepchem is required for featurize() but is not installed in this environment.")

    features = featurizer.featurize(df['SMILES'])

    if apply_filter:
        valid_ids = [i for i, f in enumerate(features) if isinstance(f, dc.feat.graph_data.GraphData)]
        features_filtered = [features[i] for i in valid_ids]
        df_filtered = df[df.index.isin(valid_ids)]
        print("")
        print(f"Org size: {len(features)}, Filtered size: {len(features_filtered)}")
        features = features_filtered
        df = df_filtered.reset_index(drop=True)

    return np.array(features), df
