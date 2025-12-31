import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

def mol_to_fp(mol):
    """Compute RDKit fingerprint for a molecule."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def is_valid_smiles(smi):
    """Return Mol object if SMILES is valid, else None."""
    return Chem.MolFromSmiles(smi)

def max_similarity(fp, fp_list):
    """Return max Tanimoto similarity between fp and list of fingerprints."""
    if not fp_list:
        return 0.0
    return max(DataStructs.TanimotoSimilarity(fp, f) for f in fp_list)

def compute_diversity(smiles_list, radius=2, nbits=2048):
    """Compute diversity as 1 - average pairwise Tanimoto similarity."""
    mols = [is_valid_smiles(s) for s in smiles_list if is_valid_smiles(s)]
    fps = [mol_to_fp(m) for m in mols]
    n = len(fps)
    if n <= 1:
        return 0.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - np.mean(sims)

def filter_valid_unique(smiles_list):
    """Filter valid and unique SMILES"""
    valid_set = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_set.add(smi)
    return list(valid_set)