import numpy as np
import pandas as pd
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import Descriptors

from .data import has_metal

remover = SaltRemover()

def keep_largest(smile):
    # Function that identifies the largest fragment in a SMILES string and returns it as a new SMILES string
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(smile), asMols=True)
    largest = None
    largest_size = 0

    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_size:
            largest_size = size
            largest = mol
    
    return Chem.MolToSmiles(largest) if largest else None

def remove_salts(smile, remover=remover):
    smile = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(smile), dontRemoveEverything=True), isomericSmiles=True) # Chirality??
    if '.' in smile:
        smile = keep_largest(smile)
    return smile

def preprocess(df):
    # Remove salts and keep largest fragment
    df['SMILES'] = df['SMILES'].apply(remove_salts)

    # Remove lone atoms
    is_lone_atom = (~df['SMILES'].str.contains(r'\.')) & (df['SMILES'].str.fullmatch(r'[A-Z][a-z]?'))
    df = df[~is_lone_atom].reset_index(drop=True)

    # Remove metals
    df = df[~df['SMILES'].apply(has_metal)].reset_index(drop=True)

    # Log-transform concentrations, set non-positive values to NaN
    df['conc'] = df['conc'].apply(lambda x: np.log10(x) if x > 0 else np.nan)
    
    return df
