import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
from rdkit import RDLogger
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

def process_pgp_data():
    print("Loading P-glycoprotein (P-gp) dataset...")
    # Read the downloaded TDC tab dataset
    df = pd.read_csv('data/pgp_broccatelli.tab', sep='\t')
    
    print(f"\nLoaded {len(df)} molecules.")
    print("Extracting 2D Morgan Fingerprints and 3D Shape Descriptors (This may take a few minutes)...")
    
    fingerprints = []
    descriptors = []
    valid_indices = []
    
    # Use tqdm for progress bar since 3D embedding takes time
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row['Drug']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            # Add Hydrogens for accurate 3D geometry
            mol = Chem.AddHs(mol)
            
            # Generate 3D Conformer
            # maxAttempts and ETKDGv3 improve convergence for macrocycles like Digoxin
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            res = AllChem.EmbedMolecule(mol, params)
            
            if res != 0:
                # Fallback to basic embedding if ETKDGv3 fails
                res = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=100)
            
            if res == 0:
                # 1. Morgan Fingerprint (2048 bits)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = np.zeros((0,), dtype=np.int8)
                Chem.DataStructs.ConvertToNumpyArray(fp, arr)
                fingerprints.append(arr.tolist())
                
                # 2. Key 2D Descriptors & 3D Shape Descriptors (Correlate with P-gp binding)
                desc = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol),
                    # 3D Shape Descriptors
                    Descriptors3D.RadiusOfGyration(mol),
                    Descriptors3D.Asphericity(mol),
                    Descriptors3D.Eccentricity(mol),
                    Descriptors3D.SpherocityIndex(mol),
                    Descriptors3D.NPR1(mol),
                    Descriptors3D.NPR2(mol)
                ]
                descriptors.append(desc)
                valid_indices.append(idx)
            
    df_valid = df.loc[valid_indices].copy()
    
    print(f"\nSuccessfully generated 3D conformers for {len(df_valid)}/{len(df)} molecules.")
    print("Normalizing descriptors...")
    scaler = StandardScaler()
    scaled_desc = scaler.fit_transform(descriptors)
    
    metadata = {
        'means': scaler.mean_.tolist(),
        'stds': scaler.scale_.tolist()
    }
    with open('models/pgp_scaler_metadata.json', 'w') as f:
        json.dump(metadata, f)
    print("Saved scaler metadata to models/pgp_scaler_metadata.json")
    
    fp_cols = [f'FP_{i}' for i in range(2048)]
    desc_cols = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds', 
                 'RadiusOfGyration', 'Asphericity', 'Eccentricity', 'SpherocityIndex', 'NPR1', 'NPR2']
    
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
