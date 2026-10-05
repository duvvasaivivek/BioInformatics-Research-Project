import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def process_pgp_data():
    print("Loading P-glycoprotein (P-gp) dataset...")
    # Read the downloaded TDC tab dataset
    df = pd.read_csv('data/pgp_broccatelli.tab', sep='\t')
    
    print(f"\nLoaded {len(df)} molecules.")
    print("Extracting 2D Morgan Fingerprints and Descriptors...")
    
    fingerprints = []
    descriptors = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = row['Drug']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            # 1. Morgan Fingerprint (2048 bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((0,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr.tolist())
            
            # 2. Key Descriptors (Correlate with P-gp binding)
            desc = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol)
            ]
            descriptors.append(desc)
            valid_indices.append(idx)
            
    df_valid = df.loc[valid_indices].copy()
    
    print("Normalizing 2D descriptors...")
    scaler = StandardScaler()
    scaled_desc = scaler.fit_transform(descriptors)
    
    fp_cols = [f'FP_{i}' for i in range(2048)]
    desc_cols = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
    
    fp_df = pd.DataFrame(fingerprints, columns=fp_cols, index=df_valid.index)
    desc_df = pd.DataFrame(scaled_desc, columns=desc_cols, index=df_valid.index)
    
    # TDC uses 'Drug' for SMILES and 'Y' for Label (1 = P-gp Substrate, 0 = Non-Substrate)
    df_valid.rename(columns={'Drug': 'SMILES', 'Y': 'Label'}, inplace=True)
    df_valid = df_valid[['SMILES', 'Label']]
    
    df_final = pd.concat([df_valid, fp_df, desc_df], axis=1)
    
    output_path = "data/processed_pgp_data.csv"
    df_final.to_csv(output_path, index=False)
    print(f"\nFeature engineering complete! Saved {df_final.shape[0]} molecules to {output_path}")

if __name__ == "__main__":
    process_pgp_data()
