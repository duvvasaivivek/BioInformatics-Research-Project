import sys
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D
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

# Check if SHAP is available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ANSI Color Codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

def load_scaler_metadata(file_path, default_means=None, default_stds=None):
    """Loads feature scaler normalization metadata from a JSON file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data['means'], data['stds']
        except Exception as e:
            print(f"[WARN] Could not load {file_path}: {e}")
            
    if default_means is not None and default_stds is not None:
        return default_means, default_stds
    return None, None

def extract_features_bbb(smiles):
    """Extracts 2048-bit Morgan Fingerprint and 6 continuous 2D physicochemical descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
        
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(list(fp.ToBitString())).astype(int)
    
    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    
    raw_desc = {
        'MolWt': mol_wt,
        'LogP': logp,
        'TPSA': tpsa,
        'HDonors': h_donors,
        'HAcceptors': h_acceptors,
        'RotBonds': rot_bonds
    }
    
    desc_list = [mol_wt, logp, tpsa, h_donors, h_acceptors, rot_bonds]
    return (fp_array, desc_list), raw_desc

def extract_features_pgp(smiles):
    """Extracts 2048-bit Morgan Fingerprint, 6 2D descriptors, and 6 3D shape conformer descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
        
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    
    if res != 0:
        res = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=100)
        
    if res != 0:
        return None, None # Embedding failed, cannot calculate 3D descriptors
        
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(list(fp.ToBitString())).astype(int)
    
    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    
    # 3D Shape Descriptors
    rg = Descriptors3D.RadiusOfGyration(mol)
    asph = Descriptors3D.Asphericity(mol)
    ecc = Descriptors3D.Eccentricity(mol)
    sph = Descriptors3D.SpherocityIndex(mol)
    npr1 = Descriptors3D.NPR1(mol)
    npr2 = Descriptors3D.NPR2(mol)
    
    raw_desc = {
        'MolWt': mol_wt,
        'LogP': logp,
        'TPSA': tpsa,
        'HDonors': h_donors,
        'HAcceptors': h_acceptors,
        'RotBonds': rot_bonds,
        'RadiusOfGyration': rg,
        'Asphericity': asph,
        'Eccentricity': ecc,
        'SpherocityIndex': sph,
        'NPR1': npr1,
        'NPR2': npr2
    }
    
    desc_list = [mol_wt, logp, tpsa, h_donors, h_acceptors, rot_bonds, rg, asph, ecc, sph, npr1, npr2]
    return (fp_array, desc_list), raw_desc

def prepare_dataframe(fp_array, continuous_features, means, stds, is_pgp=False):
    """Normalizes continuous features and constructs model input DataFrame."""
    scaled = [(val - means[i]) / (stds[i] if stds[i] != 0 else 1.0) for i, val in enumerate(continuous_features)]
    
    features = {}
    for i in range(2048):
        features[f'FP_{i}'] = fp_array[i]
        
    if is_pgp:
        desc_names = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds',
                      'RadiusOfGyration', 'Asphericity', 'Eccentricity', 'SpherocityIndex', 'NPR1', 'NPR2']
    else:
        desc_names = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
        
    for i, name in enumerate(desc_names):
        features[name] = scaled[i]
        
    return pd.DataFrame([features])

def compute_explainability(model, df, is_pgp=False):
    """Computes feature attribution either via TreeSHAP or internal model feature importances."""
    desc_names = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
    if is_pgp:
        desc_names += ['RadiusOfGyration', 'Asphericity', 'Eccentricity', 'SpherocityIndex', 'NPR1', 'NPR2']

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(df)
            vals = shap_values.values[0]
            series = pd.Series(vals, index=df.columns)
            return series, True
        except Exception:
            pass

    # Fallback to feature importance weighting * standardized score
    try:
        booster = model.get_booster()
        score_dict = booster.get_score(importance_type='weight')
        contribs = {}
        for col in df.columns:
            weight = score_dict.get(col, 0)
            contribs[col] = float(df[col].iloc[0]) * (weight / 100.0)
        return pd.Series(contribs), False
    except Exception:
        return None, False

def print_molecular_explainability(smiles, raw_bbb_desc, raw_pgp_desc, bbb_df, pgp_df, 
                                  bbb_model, pgp_model, bbb_pred, bbb_prob, pgp_pred, pgp_prob):
    """
    Displays an in-depth pharmacological explainability report
    grounded in Löscher & Potschka (2005) and Seelig (1998) biophysical rules.
    """
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}=== DETAILED MOLECULAR EXPLAINABILITY & BIOPHYSICAL ANALYSIS ==={Colors.ENDC}")
    print(f"{Colors.BLUE}Candidate Molecule SMILES: {smiles}{Colors.ENDC}\n")

    # 1. Biophysical Property Profile Table
    print(f"{Colors.BOLD}{'Property':<20} | {'Raw Value':<10} | {'Target Range (CNS Permeant & P-gp Non-Substrate)':<48}{Colors.ENDC}")
    print("-" * 85)
    
    rules = [
        ('Molecular Weight', f"{raw_pgp_desc['MolWt']:.1f} Da", "< 400 Da (P-gp substrates are usually > 450-500 Da)", raw_pgp_desc['MolWt'] < 400),
        ('Lipophilicity (LogP)', f"{raw_pgp_desc['LogP']:.2f}", "1.5 to 3.0 (Crosses membrane, avoids P-gp hydrophobic cleft)", 1.5 <= raw_pgp_desc['LogP'] <= 3.2),
        ('Polar Area (TPSA)', f"{raw_pgp_desc['TPSA']:.1f} A^2", "< 70-90 A^2 (Enables passive diffusion into lipid bilayer)", raw_pgp_desc['TPSA'] < 90),
        ('H-Bond Donors', f"{raw_pgp_desc['HDonors']}", "<= 2 (Seelig Rule: > 2 donors anchor to P-gp TM helices)", raw_pgp_desc['HDonors'] <= 2),
        ('H-Bond Acceptors', f"{raw_pgp_desc['HAcceptors']}", "<= 6 (High H-bonding creates membrane desolvation penalty)", raw_pgp_desc['HAcceptors'] <= 6),
        ('Rotatable Bonds', f"{raw_pgp_desc['RotBonds']}", "<= 5-6 (Rigidity reduces induced-fit in P-gp flexible pocket)", raw_pgp_desc['RotBonds'] <= 6),
        ('Radius of Gyration', f"{raw_pgp_desc['RadiusOfGyration']:.2f} A", "< 4.0 A (Spherical compactness evades multisite cavity)", raw_pgp_desc['RadiusOfGyration'] < 4.0),
        ('Spherocity Index', f"{raw_pgp_desc['SpherocityIndex']:.3f}", "> 0.25 (Spherical geometries favor passive membrane diffusion)", raw_pgp_desc['SpherocityIndex'] > 0.25)
    ]
    
    for name, val, target, passed in rules:
        status_icon = f"{Colors.GREEN}[PASS]{Colors.ENDC}" if passed else f"{Colors.YELLOW}[WATCH]{Colors.ENDC}"
        print(f"{name:<20} | {val:<10} | {target:<48} | {status_icon}")

    # 2. P-gp Feature Attribution (TreeSHAP or Model Attribution)
    pgp_shap, is_shap = compute_explainability(pgp_model, pgp_df, is_pgp=True)
    if pgp_shap is not None:
        method_label = "TreeSHAP Values" if is_shap else "Normalized Weight Attributions"
        print(f"\n{Colors.BOLD}--- Feature Attribution on P-gp Efflux Classifier ({method_label}) ---{Colors.ENDC}")
        continuous_cols = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds',
                           'RadiusOfGyration', 'Asphericity', 'Eccentricity', 'SpherocityIndex', 'NPR1', 'NPR2']
        
        for col in continuous_cols:
            if col in pgp_shap.index:
                impact = pgp_shap[col]
                direction = f"{Colors.RED}Increases Efflux (P-gp+ Substrate){Colors.ENDC}" if impact > 0 else f"{Colors.GREEN}Evades Efflux (P-gp- Non-Substrate){Colors.ENDC}"
                print(f" * {col:<18}: Impact = {impact:+.4f} -> {direction}")

    # 3. Pharmacological Synthesis (Referencing Loscher & Potschka 2005)
    print(f"\n{Colors.BOLD}--- Pharmacological Conclusion (Loscher & Potschka, 2005 Reference) ---{Colors.ENDC}")
    if bbb_pred == 1 and pgp_pred == 0:
        print(f"{Colors.GREEN}* [OPTIMAL CNS DRUG CANDIDATE]:{Colors.ENDC}")
        print("  This compound possesses adequate lipid solubility to passively partition across the brain endothelial capillary membrane (BBB+),")
        print("  while its low molecular volume / restricted H-donor profile prevents recognition by the P-gp (ABCB1) ATPase pump.")
        print("  Like Levetiracetam (Loscher & Potschka p. 92), it bypasses pharmacoresistance mechanisms and maintains therapeutic brain concentrations.")
    elif bbb_pred == 1 and pgp_pred == 1:
        print(f"{Colors.YELLOW}* [PARADOXICAL DRUG - CLINICAL PITFALL]:{Colors.ENDC}")
        print("  Although this molecule has positive lipophilic membrane affinity (BBB+), its bulky 3D conformation or pharmacophoric pattern")
        print("  triggers high-affinity binding to P-gp transmembrane helices (P-gp+).")
        print("  As documented for Loperamide (Table 1, p. 87), it will be rapidly extruded back into the blood stream lumen, severely blunting CNS exposure.")
    else:
        print(f"{Colors.RED}* [PERIPHERAL NON-CNS AGENT]:{Colors.ENDC}")
        print("  This compound fails passive BBB permeability (BBB-), likely due to high polar surface area, unfavorable charge, or low LogP.")
        print("  Its action will be strictly confined to peripheral tissues.")
    print("-" * 85 + "\n")

def run_benchmark_test(bbb_model, pgp_model, bbb_means, bbb_stds, pgp_means, pgp_stds):
    """Runs a validated test set of clinically characterized CNS and non-CNS reference molecules."""
    benchmarks = [
        ("Levetiracetam (Keppra)", "CC[C@@H](C(=O)N)N1CCCC1=O", "Refractory AED that bypasses P-gp (Loscher p. 92)"),
        ("Loperamide (Imodium)", "CN(C)C(=O)CC(c1ccccc1)(c2ccccc2)CCN3CCC(CC3)(c4ccc(cc4)Cl)O", "Paradoxical BBB+ but P-gp+ effluxed (Table 1)"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Classic CNS active stimulant (BBB+ / P-gp-)"),
        ("Diazepam (Valium)", "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13", "Lipophilic sedative; penetrates brain readily"),
        ("Penicillin G", "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C", "Peripheral antibiotic; poor BBB penetrant (BBB-)")
    ]

    print(f"\n{Colors.BOLD}{Colors.HEADER}========================================================================{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}     BENCHMARK VALIDATION SUITE: CLINICAL REFERENCE DRUGS (LÖSCHER 2005) {Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}========================================================================{Colors.ENDC}\n")

    for drug_name, smiles, note in benchmarks:
        print(f"{Colors.CYAN}{Colors.BOLD}--- Testing Drug: {drug_name} ---{Colors.ENDC}")
        print(f"Notes: {note}")
        print(f"SMILES: {smiles}")

        bbb_feats, raw_bbb = extract_features_bbb(smiles)
        pgp_feats, raw_pgp = extract_features_pgp(smiles)

        if bbb_feats is None or pgp_feats is None:
            print(f"{Colors.RED}Error extracting descriptors for {drug_name}{Colors.ENDC}\n")
            continue

        bbb_df = prepare_dataframe(bbb_feats[0], bbb_feats[1], bbb_means, bbb_stds, is_pgp=False)
        pgp_df = prepare_dataframe(pgp_feats[0], pgp_feats[1], pgp_means, pgp_stds, is_pgp=True)

        bbb_prob = bbb_model.predict_proba(bbb_df)[0][1]
        bbb_pred = int(bbb_prob > 0.5)

        pgp_prob = pgp_model.predict_proba(pgp_df)[0][1]
        pgp_pred = int(pgp_prob > 0.5)

        bbb_str = f"{Colors.GREEN}BBB+ ({bbb_prob*100:.1f}%){Colors.ENDC}" if bbb_pred == 1 else f"{Colors.RED}BBB- ({(1-bbb_prob)*100:.1f}%){Colors.ENDC}"
        pgp_str = f"{Colors.RED}Substrate ({pgp_prob*100:.1f}%){Colors.ENDC}" if pgp_pred == 1 else f"{Colors.GREEN}Non-Substrate ({(1-pgp_prob)*100:.1f}%){Colors.ENDC}"

        if bbb_pred == 1 and pgp_pred == 0:
            conclusion = f"{Colors.GREEN}[SUCCESS] Effective CNS Drug{Colors.ENDC}"
        elif bbb_pred == 1 and pgp_pred == 1:
            conclusion = f"{Colors.YELLOW}[WARNING] Paradoxical Drug (Effluxed){Colors.ENDC}"
        else:
            conclusion = f"{Colors.RED}[FAIL] Non-CNS Drug{Colors.ENDC}"

        print(f"Results: BBB = {bbb_str} | P-gp = {pgp_str} | Category: {conclusion}\n")

def main():
    parser = argparse.ArgumentParser(description="CNS Drug Predictor: Dual-Barrier BBB Permeability & P-gp Efflux")
    parser.add_argument("--smiles", type=str, help="Single SMILES string to analyze")
    parser.add_argument("--explain", action="store_true", help="Display full biophysical and feature attribution explainability report")
    parser.add_argument("--test-benchmark", action="store_true", help="Execute the clinical reference benchmark suite")
    args = parser.parse_args()

    print(f"\n{Colors.CYAN}{Colors.BOLD}=================================================={Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}   CNS Drug Predictor (BBB + P-gp Efflux & XAI)   {Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}=================================================={Colors.ENDC}")
    
    # 1. Load BBB Model
    bbb_model_path = os.path.join(MODELS_DIR, "bbb_xgboost_optuna_best.json")
    if not os.path.exists(bbb_model_path):
        bbb_model_path = os.path.join(MODELS_DIR, "bbb_xgboost_model.json")
        
    bbb_model = xgb.XGBClassifier()
    bbb_model.load_model(bbb_model_path)
    
    # Load BBB Scaler Metadata with fallback
    default_bbb_means = [385.4589, 2.3559, 87.0098, 2.0102, 5.3007, 4.7708]
    default_bbb_stds = [170.8765, 2.1956, 67.6848, 2.2626, 3.6692, 3.6367]
    bbb_scaler_path = os.path.join(MODELS_DIR, "bbb_scaler_metadata.json")
    bbb_means, bbb_stds = load_scaler_metadata(bbb_scaler_path, default_bbb_means, default_bbb_stds)
    
    # 2. Load P-gp Model
    pgp_model_path = os.path.join(MODELS_DIR, "pgp_xgboost_model.json")
    pgp_model = xgb.XGBClassifier()
    pgp_model.load_model(pgp_model_path)
    
    pgp_scaler_path = os.path.join(MODELS_DIR, "pgp_scaler_metadata.json")
    pgp_means, pgp_stds = load_scaler_metadata(pgp_scaler_path)
    if pgp_means is None:
        print(f"{Colors.RED}[ERROR] Could not load P-gp scaler metadata.{Colors.ENDC}")
        return
        
    print(f"{Colors.GREEN}Models and metadata loaded successfully!{Colors.ENDC}")
    if SHAP_AVAILABLE:
        print(f"{Colors.GREEN}SHAP Engine: Enabled (TreeSHAP active){Colors.ENDC}\n")
    else:
        print(f"{Colors.YELLOW}SHAP Engine: Using internal model attribution (run 'pip install shap' for full TreeSHAP){Colors.ENDC}\n")

    # If --test-benchmark is requested
    if args.test_benchmark:
        run_benchmark_test(bbb_model, pgp_model, bbb_means, bbb_stds, pgp_means, pgp_stds)
        return

    # If a SMILES was provided via CLI
    if args.smiles:
        candidate_list = [args.smiles]
        interactive = False
    else:
        interactive = True

    while True:
        try:
            if interactive:
                try:
                    user_input = input(f"{Colors.BOLD}Enter a SMILES string (or 'bench' for benchmark, 'exit' to quit):{Colors.ENDC} ").strip()
                except EOFError:
                    break
                    
                if user_input.lower() in ['exit', 'quit']:
                    print(f"{Colors.YELLOW}Exiting...{Colors.ENDC}")
                    break
                if user_input.lower() == 'bench':
                    run_benchmark_test(bbb_model, pgp_model, bbb_means, bbb_stds, pgp_means, pgp_stds)
                    continue
                if not user_input:
                    continue
            else:
                user_input = candidate_list.pop(0)

            # Feature Extraction
            bbb_feats, raw_bbb_desc = extract_features_bbb(user_input)
            pgp_feats, raw_pgp_desc = extract_features_pgp(user_input)
            
            if bbb_feats is None:
                print(f"{Colors.RED}[ERROR] Invalid SMILES string: {user_input}{Colors.ENDC}")
                if not interactive: break
                continue
            if pgp_feats is None:
                print(f"{Colors.RED}[ERROR] Could not generate 3D conformer for: {user_input}{Colors.ENDC}")
                if not interactive: break
                continue
                
            bbb_fp, bbb_desc = bbb_feats
            pgp_fp, pgp_desc = pgp_feats
            
            # Predict BBB Permeability
            bbb_df = prepare_dataframe(bbb_fp, bbb_desc, bbb_means, bbb_stds, is_pgp=False)
            bbb_prob = bbb_model.predict_proba(bbb_df)[0][1]
            bbb_pred = int(bbb_prob > 0.5)
            
            # Predict P-gp Substrate
            pgp_df = prepare_dataframe(pgp_fp, pgp_desc, pgp_means, pgp_stds, is_pgp=True)
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
            print(f"{Colors.CYAN}--------------------------------------------------{Colors.ENDC}")

            # Explainability Report (always shown if --explain or interactive)
            print_molecular_explainability(user_input, raw_bbb_desc, raw_pgp_desc, bbb_df, pgp_df, 
                                           bbb_model, pgp_model, bbb_pred, bbb_prob, pgp_pred, pgp_prob)
            
            if not interactive:
                break
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Exiting...{Colors.ENDC}")
            break

if __name__ == "__main__":
    main()
