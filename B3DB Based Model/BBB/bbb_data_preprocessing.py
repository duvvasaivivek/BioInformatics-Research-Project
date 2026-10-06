import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from sklearn.preprocessing import StandardScaler
import os

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

DATA_PATH = "../data/B3DB_classification.tsv"
OUTPUT_PATH = "../data/processed_b3db_data_advanced.csv"

def process_data():
    print(f"Loading dataset from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, sep='\t')
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}.")
        return

    df = df[['SMILES', 'BBB+/BBB-']].dropna()
    df['Label'] = df['BBB+/BBB-'].map({'BBB+': 1, 'BBB-': 0})
    df = df.drop('BBB+/BBB-', axis=1)
    
    print("Generating Morgan Fingerprints and 2D Descriptors...")
    fingerprints = []
    descriptors = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            # 1. Fingerprint (2048 bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((0,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr.tolist())
            
            # 2. 2D Physicochemical Descriptors (6 continuous features)
            desc = [
                Descriptors.MolWt(mol),             # Molecular Weight
                Descriptors.MolLogP(mol),           # Lipophilicity
                Descriptors.TPSA(mol),              # Topological Polar Surface Area
                Descriptors.NumHDonors(mol),        # Hydrogen Bond Donors
                Descriptors.NumHAcceptors(mol),     # Hydrogen Bond Acceptors
                Descriptors.NumRotatableBonds(mol)  # Flexibility
            ]
            descriptors.append(desc)
            valid_indices.append(idx)
            
    df_valid = df.loc[valid_indices].copy()
    
    # Scale descriptors (Standardization) so large numbers like MolWt 
    # don't overshadow the small 0/1 fingerprint values
    print("Normalizing 2D descriptors...")
    scaler = StandardScaler()
    scaled_desc = scaler.fit_transform(descriptors)
    
    # Create columns
    fp_cols = [f'FP_{i}' for i in range(2048)]
    desc_cols = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
    
    # Combine everything into one giant Dataframe
    fp_df = pd.DataFrame(fingerprints, columns=fp_cols, index=df_valid.index)
    desc_df = pd.DataFrame(scaled_desc, columns=desc_cols, index=df_valid.index)
    
    df_final = pd.concat([df_valid, fp_df, desc_df], axis=1)
    
    print(f"Feature engineering complete. Final dataset shape: {df_final.shape}")
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_data()
