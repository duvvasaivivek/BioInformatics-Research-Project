"""
Core Feature Extraction and Preprocessing Module for Novelty 2 (Sequential Sieve & Transfer).
Centralizes 2D physicochemical descriptor calculation, RDKit ETKDGv3 3D conformer embedding,
Morgan fingerprint generation, and standardized input preparation for Stage 1 (BBB Sieve)
and Stage 2 (P-gp Gatekeeper).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdFingerprintGenerator
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

BBB_DESCRIPTOR_NAMES = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
PGP_DESCRIPTOR_NAMES = [
    'MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds',
    'RadiusOfGyration', 'Asphericity', 'Eccentricity', 'SpherocityIndex', 'NPR1', 'NPR2'
]

_MORGAN_GEN_R2_2048 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def load_scaler_metadata(file_path: str) -> Tuple[List[float], List[float]]:
    """Loads and validates scaler metadata for descriptor z-scoring."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Scaler metadata not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    names = PGP_DESCRIPTOR_NAMES if 'pgp' in os.path.basename(file_path).lower() else BBB_DESCRIPTOR_NAMES
    means = np.asarray(data['means'], dtype=float)
    stds = np.asarray(data['stds'], dtype=float)
    if data.get('features') != names or len(means) != len(names) or len(stds) != len(names):
        raise ValueError(f"Scaler feature schema mismatch in {file_path}")
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("Scaler statistics must be finite with strictly positive standard deviations")
    return means.tolist(), stds.tolist()


def calculate_2d_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Calculates the 6 core 2D physicochemical descriptors for Stage 1 BBB screening."""
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
    Generates a 3D conformer via RDKit ETKDGv3 and extracts
    six stereochemical shape descriptors for Stage 2 P-gp efflux screening.
    Only invoked when a molecule passes the Stage 1 BBB sieve.
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
    """Calculates 2048-bit Morgan circular fingerprint (ECFP4 equivalent) on heavy atoms."""
    if radius == 2 and n_bits == 2048:
        fp = _MORGAN_GEN_R2_2048.GetFingerprint(mol)
    else:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
    arr = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(int)


def extract_features_bbb(smiles: str) -> Tuple[Optional[Tuple[np.ndarray, List[float]]], Optional[Dict[str, float]], Optional[Chem.Mol]]:
    """
    Extracts 2048-bit Morgan Fingerprint and 6 continuous 2D physicochemical descriptors for Stage 1 BBB.
    Fast screening phase without conformer overhead.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    fp = calculate_morgan_fingerprint(mol)
    desc_dict = calculate_2d_descriptors(mol)
    desc_list = [desc_dict[name] for name in BBB_DESCRIPTOR_NAMES]
    return (fp, desc_list), desc_dict, mol


def extract_features_pgp(smiles: str, seed: int = 42) -> Tuple[Optional[Tuple[np.ndarray, List[float]]], Optional[Dict[str, float]], Optional[Chem.Mol]]:
    """
    Extracts 2048-bit Morgan Fingerprint, 6 2D descriptors, and 6 3D shape descriptors for Stage 2 P-gp.
    Only executed sequentially for compounds that pass the Stage 1 sieve.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    fp = calculate_morgan_fingerprint(mol)
    desc_2d = calculate_2d_descriptors(mol)
    desc_3d, mol_3d = calculate_3d_descriptors(mol, seed=seed)
    if desc_3d is None:
        return None, None, None
    full_desc = {**desc_2d, **desc_3d}
    desc_list = [full_desc[name] for name in PGP_DESCRIPTOR_NAMES]
    return (fp, desc_list), full_desc, mol_3d


def prepare_dataframe(fp: np.ndarray, desc: List[float], means: List[float], stds: List[float], is_pgp: bool = False) -> pd.DataFrame:
    """Prepares standard aligned dataframe with z-score scaled continuous descriptors."""
    cols = [f"FP_{i}" for i in range(len(fp))]
    desc_names = PGP_DESCRIPTOR_NAMES if is_pgp else BBB_DESCRIPTOR_NAMES
    cols.extend(desc_names)
    scaled_desc = [(v - m) / s for v, m, s in zip(desc, means, stds)]
    row = np.hstack([fp, scaled_desc]).reshape(1, -1)
    return pd.DataFrame(row, columns=cols)
