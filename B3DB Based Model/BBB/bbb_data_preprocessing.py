"""
BBB Data Preprocessing Pipeline.
Processes raw B3DB classification dataset, generates 2048-bit Morgan Fingerprints
and standardized 2D physicochemical descriptors using core feature extraction routines,
and serializes scaler metadata for deterministic inference.
"""

import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit C++ warnings
RDLogger.DisableLog('rdApp.*')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.features import (
    BBB_DESCRIPTOR_NAMES,
    calculate_2d_descriptors,
    calculate_morgan_fingerprint
)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(DATA_DIR, "B3DB_classification.tsv")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_b3db_data_advanced.csv")


def process_data():
    print(f"Loading BBB dataset from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find {DATA_PATH}.")
        return

    df = pd.read_csv(DATA_PATH, sep='\t')
    df = df[['SMILES', 'BBB+/BBB-']].dropna()
    df['Label'] = df['BBB+/BBB-'].map({'BBB+': 1, 'BBB-': 0})
    df = df.drop('BBB+/BBB-', axis=1)

    print("Generating Morgan Fingerprints (2048 bits) and 2D Descriptors...")
    fingerprints = []
    descriptors = []
    valid_indices = []

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            fp_arr = calculate_morgan_fingerprint(mol, radius=2, n_bits=2048)
            desc_dict = calculate_2d_descriptors(mol)

            fingerprints.append(fp_arr.tolist())
            descriptors.append([desc_dict[k] for k in BBB_DESCRIPTOR_NAMES])
            valid_indices.append(idx)

    df_valid = df.loc[valid_indices].copy()

    fp_cols = [f'FP_{i}' for i in range(2048)]
    fp_df = pd.DataFrame(fingerprints, columns=fp_cols, index=df_valid.index)
    desc_df = pd.DataFrame(descriptors, columns=BBB_DESCRIPTOR_NAMES, index=df_valid.index)

    df_final = pd.concat([df_valid, fp_df, desc_df], axis=1)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature engineering complete. Saved {df_final.shape[0]} molecules with raw descriptors to {OUTPUT_PATH}")


if __name__ == "__main__":
    process_data()
