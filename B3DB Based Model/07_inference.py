import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def smiles_to_features(smiles):
    """Converts a SMILES string into the exact 2054-dimensional feature vector used during training."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"\n[ERROR] Invalid SMILES string: {smiles}. RDKit could not parse it.")
        return None
    
    # 1. Morgan Fingerprint (2048 bits, radius 2)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(list(fp.ToBitString())).astype(int)
    
    # 2. Continuous Descriptors (6 key physicochemical properties)
    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    
    # Standard Scaler Means and Stds from training data (B3DB)
    means = [385.4589, 2.3559, 87.0098, 2.0102, 5.3007, 4.7708]
    stds = [170.8765, 2.1956, 67.6848, 2.2626, 3.6692, 3.6367]
    
    # Standardize them: (Value - Mean) / Std
    mol_wt = (mol_wt - means[0]) / stds[0]
    logp = (logp - means[1]) / stds[1]
    tpsa = (tpsa - means[2]) / stds[2]
    h_donors = (h_donors - means[3]) / stds[3]
    h_acceptors = (h_acceptors - means[4]) / stds[4]
    rot_bonds = (rot_bonds - means[5]) / stds[5]
    
    # 3. Combine into a single dictionary matching the exact column names of the training data
    features = {}
    for i in range(2048):
        features[f'FP_{i}'] = fp_array[i]
        
    features['MolWt'] = mol_wt
    features['LogP'] = logp
    features['TPSA'] = tpsa
    features['HDonors'] = h_donors
    features['HAcceptors'] = h_acceptors
    features['RotBonds'] = rot_bonds
    
    # Create DataFrame (1 row, 2054 columns)
    df = pd.DataFrame([features])
    return df

def main():
    print("==================================================")
    print("      Blood-Brain Barrier (BBB) Predictor         ")
    print("==================================================")
    
    # Load the absolute best trained model
    model_path = "bbb_xgboost_optuna_best.json"
    print(f"Loading Model: {model_path}")
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return
        
    print("Model loaded successfully! Type 'exit' to quit.\n")
    
    # Interactive inference loop
    while True:
        try:
            user_input = input("Enter a SMILES string: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            if not user_input:
                continue
                
            features_df = smiles_to_features(user_input)
            if features_df is None:
                continue
                
            # Make the prediction!
            # predict_proba returns [[prob_class_0, prob_class_1]]
            prob = model.predict_proba(features_df)[0][1] 
            prediction = int(prob > 0.5)
            
            print(f"\n[RESULTS FOR]: {user_input}")
            if prediction == 1:
                print(f"► Prediction:  BBB+ (WILL cross the Blood-Brain Barrier)")
            else:
                print(f"► Prediction:  BBB- (Will NOT cross the Blood-Brain Barrier)")
                
            print(f"► Confidence:  {max(prob, 1-prob)*100:.2f}%")
            print("--------------------------------------------------\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
if __name__ == "__main__":
    main()
