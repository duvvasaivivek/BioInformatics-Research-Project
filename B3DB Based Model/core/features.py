"""
Core Feature Extraction and Preprocessing Module for Dual-Barrier CNS Drug Predictor.
Centralizes 2D physicochemical descriptor calculation, RDKit ETKDGv3 3D conformer embedding,
Morgan fingerprint generation, and standardized input preparation for BBB and P-gp models.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from rdkit import RDLogger

# Suppress RDKit C++ warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

BBB_DESCRIPTOR_NAMES = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
PGP_DESCRIPTOR_NAMES = [
    'MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds',
    'RadiusOfGyration', 'Asphericity', 'Eccentricity', 'SpherocityIndex', 'NPR1', 'NPR2'
]


def load_scaler_metadata(
    file_path: str,
    default_means: Optional[List[float]] = None,
    default_stds: Optional[List[float]] = None
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Loads feature scaler normalization metadata from a JSON file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data['means'], data['stds']
        except Exception as e:
            print(f"[WARN] Could not load scaler metadata from {file_path}: {e}")

    if default_means is not None and default_stds is not None:
        return default_means, default_stds
    return None, None


def calculate_2d_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Calculates the 6 core 2D physicochemical descriptors."""
    return {
        'MolWt': float(Descriptors.MolWt(mol)),
        'LogP': float(Descriptors.MolLogP(mol)),
        'TPSA': float(Descriptors.TPSA(mol)),
        'HDonors': float(Descriptors.NumHDonors(mol)),
        'HAcceptors': float(Descriptors.NumHAcceptors(mol)),
        'RotBonds': float(Descriptors.NumRotatableBonds(mol))
    }


def calculate_3d_descriptors(mol: Chem.Mol, seed: int = 42) -> Tuple[Optional[Dict[str, float]], Optional[Chem.Mol]]:
    """
    Generates a 3D conformer via RDKit ETKDGv3 (with fallback) and extracts
    six stereochemical shape descriptors for P-gp efflux modeling.
    """
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    res = AllChem.EmbedMolecule(mol_h, params)

    if res != 0:
        res = AllChem.EmbedMolecule(mol_h, randomSeed=seed, maxAttempts=100)

    if res != 0:
        return None, None

    desc_3d = {
        'RadiusOfGyration': float(Descriptors3D.RadiusOfGyration(mol_h)),
        'Asphericity': float(Descriptors3D.Asphericity(mol_h)),
        'Eccentricity': float(Descriptors3D.Eccentricity(mol_h)),
        'SpherocityIndex': float(Descriptors3D.SpherocityIndex(mol_h)),
        'NPR1': float(Descriptors3D.NPR1(mol_h)),
        'NPR2': float(Descriptors3D.NPR2(mol_h))
    }
    return desc_3d, mol_h


def calculate_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Calculates Morgan circular fingerprint as an integer numpy bit array."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(int)


def extract_features_bbb(smiles: str) -> Tuple[Optional[Tuple[np.ndarray, List[float]]], Optional[Dict[str, float]], Optional[Chem.Mol]]:
    """
    Extracts 2048-bit Morgan Fingerprint and 6 continuous 2D physicochemical descriptors for BBB modeling.
    Returns: ((fp_array, desc_list), raw_desc_dict, mol)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    fp_array = calculate_morgan_fingerprint(mol, radius=2, n_bits=2048)
    raw_desc = calculate_2d_descriptors(mol)
    desc_list = [raw_desc[k] for k in BBB_DESCRIPTOR_NAMES]

    return (fp_array, desc_list), raw_desc, mol


def extract_features_pgp(smiles: str) -> Tuple[Optional[Tuple[np.ndarray, List[float]]], Optional[Dict[str, float]], Optional[Chem.Mol]]:
    """
    Extracts 2048-bit Morgan Fingerprint, 6 2D descriptors, and 6 3D shape conformer descriptors for P-gp modeling.
    Returns: ((fp_array, desc_list), raw_desc_dict, mol_h)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    desc_3d, mol_h = calculate_3d_descriptors(mol)
    if desc_3d is None:
        return None, None, mol

    fp_array = calculate_morgan_fingerprint(mol, radius=2, n_bits=2048)
    raw_desc = calculate_2d_descriptors(mol)
    raw_desc.update(desc_3d)

    desc_list = [raw_desc[k] for k in PGP_DESCRIPTOR_NAMES]
    return (fp_array, desc_list), raw_desc, mol_h


def prepare_dataframe(
    fp_array: np.ndarray,
    continuous_features: List[float],
    means: List[float],
    stds: List[float],
    is_pgp: bool = False
) -> pd.DataFrame:
    """Normalizes continuous features and constructs a single-row DataFrame matching training column schema."""
    scaled = [
        (val - means[i]) / (stds[i] if stds[i] != 0 else 1.0)
        for i, val in enumerate(continuous_features)
    ]

    features = {}
    for i in range(len(fp_array)):
        features[f'FP_{i}'] = fp_array[i]

    desc_names = PGP_DESCRIPTOR_NAMES if is_pgp else BBB_DESCRIPTOR_NAMES
    for i, name in enumerate(desc_names):
        features[name] = scaled[i]

    return pd.DataFrame([features])
