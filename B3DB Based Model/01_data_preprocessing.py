import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import os

# Suppress RDKit warnings (like MorganGenerator deprecation)
RDLogger.DisableLog('rdApp.*')

# 1. Define Paths
# Use the classification dataset
DATA_PATH = "B3DB_classification.tsv"
OUTPUT_PATH = "processed_b3db_data.csv"

def process_data():
    print(f"Loading dataset from {DATA_PATH}...")
    
    # Load TSV Data
    try:
        df = pd.read_csv(DATA_PATH, sep='\t')
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}. Make sure Phase 1 completed successfully.")
        return

    # 2. Select Relevant Columns
    # We primarily need SMILES (chemical structure) and BBB+/BBB- (target label)
    df = df[['SMILES', 'BBB+/BBB-']]
    
    # Drop rows with missing SMILES or labels
    df = df.dropna()
    print(f"Shape after dropping missing values: {df.shape}")

    # 3. Target Label Encoding
    # Map BBB+ to 1 (permeable) and BBB- to 0 (non-permeable)
    df['Label'] = df['BBB+/BBB-'].map({'BBB+': 1, 'BBB-': 0})
    
    # Drop the original label column to keep it clean
    df = df.drop('BBB+/BBB-', axis=1)
    
    # 4. Feature Engineering (RDKit Morgan Fingerprints)
    print("Generating Morgan Fingerprints using RDKit...")
    
    fingerprints = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            # Generate Morgan Fingerprint (radius 2, 2048 bits) - standard for ML
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            # Convert RDKit vector to numpy array, then to a list of ints
            arr = np.zeros((0,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr.tolist())
            valid_indices.append(idx)
        else:
            print(f"Warning: Could not parse SMILES at index {idx}: {smiles}")
            
    # Filter dataframe to only include molecules RDKit could parse successfully
    df_valid = df.loc[valid_indices].copy()
    
    # Create column names for the 2048 fingerprint bits
    fp_cols = [f'FP_{i}' for i in range(2048)]
    
    # Create a DataFrame for fingerprints and concatenate with the main dataframe
    fp_df = pd.DataFrame(fingerprints, columns=fp_cols, index=df_valid.index)
    df_final = pd.concat([df_valid, fp_df], axis=1)
    
    print(f"Feature engineering complete. Final dataset shape: {df_final.shape}")
    
    # 5. Save the Processed Data
    print(f"Saving processed data to {OUTPUT_PATH}...")
    df_final.to_csv(OUTPUT_PATH, index=False)
    print("Done! You are ready for Phase 3 (Model Building & Training).")

if __name__ == "__main__":
    process_data()
