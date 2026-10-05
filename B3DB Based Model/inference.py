import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
import warnings
import json
import os

# Enable ANSI escape sequences on Windows
os.system('color')

# Suppress Python warnings
warnings.filterwarnings('ignore')
# Suppress RDKit C++ warnings
RDLogger.DisableLog('rdApp.*')

# ANSI Color Codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def load_scaler_metadata(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['means'], data['stds']
    except Exception as e:
        print(f"[ERROR] Could not load {file_path}: {e}")
        return None, None

def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
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
    
    return fp_array, [mol_wt, logp, tpsa, h_donors, h_acceptors, rot_bonds]

def prepare_dataframe(fp_array, continuous_features, means, stds):
    # Standardize: (Value - Mean) / Std
    scaled = [(val - means[i]) / stds[i] for i, val in enumerate(continuous_features)]
    
    features = {}
    for i in range(2048):
        features[f'FP_{i}'] = fp_array[i]
        
    features['MolWt'] = scaled[0]
    features['LogP'] = scaled[1]
    features['TPSA'] = scaled[2]
    features['HDonors'] = scaled[3]
    features['HAcceptors'] = scaled[4]
    features['RotBonds'] = scaled[5]
    
    return pd.DataFrame([features])

def main():
    print(f"\n{Colors.CYAN}{Colors.BOLD}=================================================={Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}      CNS Drug Predictor (BBB + P-gp Efflux)      {Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}=================================================={Colors.ENDC}")
    
    # 1. Load BBB Model
    bbb_model_path = "models/bbb_xgboost_optuna_best.json"
    bbb_model = xgb.XGBClassifier()
    bbb_model.load_model(bbb_model_path)
    # BBB used hardcoded training data means/stds in previous script, we can hardcode them here to avoid needing to parse the big B3DB csv again
    bbb_means = [385.4589, 2.3559, 87.0098, 2.0102, 5.3007, 4.7708]
    bbb_stds = [170.8765, 2.1956, 67.6848, 2.2626, 3.6692, 3.6367]
    
    # 2. Load P-gp Model
    pgp_model_path = "models/pgp_xgboost_model.json"
    pgp_model = xgb.XGBClassifier()
    pgp_model.load_model(pgp_model_path)
    
    pgp_means, pgp_stds = load_scaler_metadata("models/pgp_scaler_metadata.json")
    if pgp_means is None:
        return
        
    print(f"{Colors.GREEN}Models loaded successfully! Type 'exit' to quit.{Colors.ENDC}\n")
    
    while True:
        try:
            try:
                user_input = input(f"{Colors.BOLD}Enter a SMILES string:{Colors.ENDC} ").strip()
            except EOFError:
                break
                
            if user_input.lower() in ['exit', 'quit']:
                print(f"{Colors.YELLOW}Exiting...{Colors.ENDC}")
                break
            if not user_input:
                continue
                
            features_tuple = extract_features(user_input)
            if features_tuple is None:
                print(f"[ERROR] Invalid SMILES string: {user_input}")
                continue
            
            fp_array, continuous_desc = features_tuple
            
            # Predict BBB Permeability
            bbb_df = prepare_dataframe(fp_array, continuous_desc, bbb_means, bbb_stds)
            bbb_prob = bbb_model.predict_proba(bbb_df)[0][1]
            bbb_pred = int(bbb_prob > 0.5)
            
            # Predict P-gp Substrate
            pgp_df = prepare_dataframe(fp_array, continuous_desc, pgp_means, pgp_stds)
            pgp_prob = pgp_model.predict_proba(pgp_df)[0][1]
            pgp_pred = int(pgp_prob > 0.5)
            
            print(f"\n{Colors.BLUE}[RESULTS FOR]: {user_input}{Colors.ENDC}")
            
            # BBB Output
            if bbb_pred == 1:
                print(f"-> BBB Permeability: {Colors.GREEN}BBB+ (Permeable){Colors.ENDC} (Confidence: {bbb_prob*100:.1f}%)")
            else:
                print(f"-> BBB Permeability: {Colors.RED}BBB- (Non-Permeable){Colors.ENDC} (Confidence: {(1-bbb_prob)*100:.1f}%)")
                
            # P-gp Output
            if pgp_pred == 1:
                print(f"-> P-gp Efflux Pump: {Colors.RED}Substrate (Pumped Out){Colors.ENDC} (Confidence: {pgp_prob*100:.1f}%)")
            else:
                print(f"-> P-gp Efflux Pump: {Colors.GREEN}Non-Substrate (Remains){Colors.ENDC} (Confidence: {(1-pgp_prob)*100:.1f}%)")
                
            # Summary
            print(f"\n{Colors.BOLD}-- CNS ACTION SUMMARY --{Colors.ENDC}")
            if bbb_pred == 1 and pgp_pred == 0:
                print(f"{Colors.GREEN}[SUCCESS] Effective CNS Drug: Enters the brain and is NOT pumped out by P-gp.{Colors.ENDC}")
            elif bbb_pred == 1 and pgp_pred == 1:
                print(f"{Colors.YELLOW}[WARNING] Paradoxical Drug: Enters the brain (BBB+) but is actively pumped out by P-gp (e.g. Loperamide). Poor CNS efficacy.{Colors.ENDC}")
            else:
                print(f"{Colors.RED}[FAIL] Non-CNS Drug: Does not cross the Blood-Brain Barrier (BBB-).{Colors.ENDC}")
            print(f"{Colors.CYAN}--------------------------------------------------{Colors.ENDC}\n")
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Exiting...{Colors.ENDC}")
            break
            
if __name__ == "__main__":
    main()
