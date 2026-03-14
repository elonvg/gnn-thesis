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

def salt_remover(smile, remover=remover):
    smile = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(smile), dontRemoveEverything=True), isomericSmiles=True) # Chirality??
    if '.' in smile:
        smile = keep_largest(smile)
    return smile

def preprocess(df, remove_salts=False, remove_lone=False, remove_metals=False):
    # Remove salts and keep largest fragment
    if remove_salts:
        df['SMILES'] = df['SMILES'].apply(salt_remover)

    # Remove lone atoms
    if remove_lone:
        is_lone_atom = (~df['SMILES'].str.contains(r'\.')) & (df['SMILES'].str.fullmatch(r'[A-Z][a-z]?'))
        df = df[~is_lone_atom].reset_index(drop=True)

    # Remove metals
    if remove_metals:
        df = df[~df['SMILES'].apply(has_metal)].reset_index(drop=True)

    # Convert conc to log10 
    df = preprocess_conc(df)
    
    return df

def preprocess_conc(df):

    # TODO: Handle different concentration units

    # Log-transform concentrations, set non-positive values to NaN
    df['conc'] = df['conc'].apply(lambda x: np.log10(x) if x > 0 else np.nan)
    
    return df
