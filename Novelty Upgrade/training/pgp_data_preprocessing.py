"""
P-glycoprotein (P-gp / ABCB1) Data Preprocessing Pipeline.
Processes the Broccatelli benchmark dataset, generates 2048-bit Morgan Fingerprints,
6 continuous 2D descriptors, and 6 stereochemical 3D conformer shape descriptors
using core feature extraction routines, and serializes scaler metadata.
"""

import os
import sys
import warnings
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit C++ warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.features import (
    PGP_DESCRIPTOR_NAMES,
    calculate_2d_descriptors,
    calculate_3d_descriptors,
    calculate_morgan_fingerprint
)

DATA_PATH = os.path.join(BASE_DIR, "data", "pgp_broccatelli.tab")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed_pgp_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def process_pgp_data():
    print("Loading P-glycoprotein (P-gp) dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find {DATA_PATH}.")
        return

    df = pd.read_csv(DATA_PATH, sep='\t')
    print(f"\nLoaded {len(df)} molecules.")
    print("Extracting 2D Morgan Fingerprints and 3D Shape Descriptors via ETKDGv3...")

    fingerprints = []
    descriptors = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row['Drug']
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            desc_3d, _ = calculate_3d_descriptors(mol, seed=42)

            if desc_3d is not None:
                fp_arr = calculate_morgan_fingerprint(mol, radius=2, n_bits=2048)
                desc_2d = calculate_2d_descriptors(mol)

                combined_desc = desc_2d
                combined_desc.update(desc_3d)

                fingerprints.append(fp_arr.tolist())
                descriptors.append([combined_desc[k] for k in PGP_DESCRIPTOR_NAMES])
                valid_indices.append(idx)

    df_valid = df.loc[valid_indices].copy()
    print(f"\nSuccessfully generated 3D conformers for {len(df_valid)}/{len(df)} molecules.")

    fp_cols = [f'FP_{i}' for i in range(2048)]
    fp_df = pd.DataFrame(fingerprints, columns=fp_cols, index=df_valid.index)
    desc_df = pd.DataFrame(descriptors, columns=PGP_DESCRIPTOR_NAMES, index=df_valid.index)

    df_valid.rename(columns={'Drug': 'SMILES', 'Y': 'Label'}, inplace=True)
    df_valid = df_valid[['SMILES', 'Label']]

    df_final = pd.concat([df_valid, fp_df, desc_df], axis=1)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFeature engineering complete! Saved {df_final.shape[0]} molecules with raw descriptors to {OUTPUT_PATH}")


if __name__ == "__main__":
    process_pgp_data()
