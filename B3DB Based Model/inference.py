"""
Dual-Barrier CNS Drug Predictor & Explainable AI Suite.
Decouples passive Blood-Brain Barrier (BBB) permeability from active P-glycoprotein (P-gp) efflux
using Bayesian-tuned XGBoost ensembles, 3D conformer geometry, and TreeSHAP feature attributions.
"""

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
import datetime
from typing import Optional, Dict, Any, List, Tuple

# Enable ANSI escape sequences on Windows
os.system('color')

# Suppress Python warnings
warnings.filterwarnings('ignore')
# Suppress RDKit C++ warnings
RDLogger.DisableLog('rdApp.*')

# Import Explainability Engine
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from explainability import (
    DualBarrierShapExplainer,
    MorganSubstructureMapper,
    ApplicabilityDomainEstimator,
    SharedPropertiesAnalyzer,
    MolecularCounterfactualEngine,
    ScientificReportGenerator
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


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
        return None, None, None

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
    return (fp_array, desc_list), raw_desc, mol


def extract_features_pgp(smiles):
    """Extracts 2048-bit Morgan Fingerprint, 6 2D descriptors, and 6 3D shape conformer descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol_h, params)

    if res != 0:
        res = AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=100)

    if res != 0:
        return None, None, mol  # 3D embedding failed

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(list(fp.ToBitString())).astype(int)

    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)

    # 3D Shape Descriptors
    rg = Descriptors3D.RadiusOfGyration(mol_h)
    asph = Descriptors3D.Asphericity(mol_h)
    ecc = Descriptors3D.Eccentricity(mol_h)
    sph = Descriptors3D.SpherocityIndex(mol_h)
    npr1 = Descriptors3D.NPR1(mol_h)
    npr2 = Descriptors3D.NPR2(mol_h)

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
    return (fp_array, desc_list), raw_desc, mol_h


def prepare_dataframe(fp_array, continuous_features, means, stds, is_pgp=False):
    """Normalizes continuous features and constructs model input DataFrame with exact column order."""
    scaled = [(val - means[i]) / (stds[i] if stds[i] != 0 else 1.0) for i, val in enumerate(continuous_features)]

    features = {}
    for i in range(2048):
        features[f'FP_{i}'] = fp_array[i]

    if is_pgp:
        desc_names = [
            'MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds',
            'RadiusOfGyration', 'Asphericity', 'Eccentricity', 'SpherocityIndex', 'NPR1', 'NPR2'
        ]
    else:
        desc_names = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']

    for i, name in enumerate(desc_names):
        features[name] = scaled[i]

    return pd.DataFrame([features])


def determine_dual_barrier_quadrant(bbb_pred, pgp_pred):
    """
    Classifies candidate into one of four pharmacologically grounded quadrants:
    1. BBB+ / P-gp- : Optimal CNS Drug Candidate (Passive Influx, No Pump Extrusion)
    2. BBB+ / P-gp+ : Paradoxical Drug Candidate (Lipophilic Influx, Violent Efflux - e.g. Loperamide)
    3. BBB- / P-gp- : Peripheral Low-Permeation Agent (Fails Passive Influx, Evades Pump)
    4. BBB- / P-gp+ : Restricted Effluxed Peripheral Compound (Poor Permeation & Active Clearance)
    """
    if bbb_pred == 1 and pgp_pred == 0:
        return (
            "Quadrant 1: BBB+ / P-gp- [Optimal CNS Candidate]",
            "Compound possesses favorable lipid partition to cross brain capillary endothelial membranes passively, "
            "and does not engage the ABCB1 (P-gp) ATPase pump. Like Levetiracetam, it maintains sustained brain exposure."
        )
    elif bbb_pred == 1 and pgp_pred == 1:
        return (
            "Quadrant 2: BBB+ / P-gp+ [Paradoxical Drug / High Efflux Risk]",
            "Compound readily partitions into endothelial membranes (BBB+), but high spatial bulk or Seelig H-bonding motifs "
            "trigger active luminal extrusion by P-glycoprotein. Resembles Loperamide: high in vitro lipophilicity but negligible brain exposure."
        )
    elif bbb_pred == 0 and pgp_pred == 0:
        return (
            "Quadrant 3: BBB- / P-gp- [Peripheral Low-Permeation Agent]",
            "Compound lacks sufficient lipid solubility or possesses excessive polar surface area to cross the BBB passively. "
            "Although it does not trigger P-gp efflux, lack of passive permeation prevents CNS penetration."
        )
    else:
        return (
            "Quadrant 4: BBB- / P-gp+ [Restricted Effluxed Peripheral Compound]",
            "Compound is doubly impeded: it fails passive BBB permeation and is actively recognized by P-gp as an efflux substrate. "
            "Action is strictly confined to peripheral compartments."
        )


def display_full_scientific_report(
    smiles,
    mol,
    raw_bbb_desc,
    raw_pgp_desc,
    bbb_df,
    pgp_df,
    bbb_model,
    pgp_model,
    bbb_pred,
    bbb_prob,
    pgp_pred,
    pgp_prob,
    shap_explainer: DualBarrierShapExplainer,
    shared_analyzer: SharedPropertiesAnalyzer,
    domain_estimator: ApplicabilityDomainEstimator,
    counterfactual_engine: Optional[MolecularCounterfactualEngine] = None,
    run_counterfactuals: bool = False
) -> Dict[str, Any]:
    """
    Generates and prints a comprehensive, multi-section scientific explainability report.
    """
    quadrant_name, quadrant_desc = determine_dual_barrier_quadrant(bbb_pred, pgp_pred)

    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*85}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}     DUAL-BARRIER SCIENTIFIC EXPLAINABILITY & PHARMACOLOGICAL AUDIT       {Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*85}{Colors.ENDC}")
    print(f"{Colors.BLUE}Candidate SMILES:{Colors.ENDC} {smiles}\n")

    # SECTION 1: Dual-Barrier Four-Quadrant Assessment
    print(f"{Colors.BOLD}{Colors.UNDERLINE}1. DUAL-BARRIER FOUR-QUADRANT ASSESSMENT{Colors.ENDC}")
    print(f"Classification: {Colors.BOLD}{quadrant_name}{Colors.ENDC}")
    print(f"Pharmacological Profile: {quadrant_desc}\n")

    # Scaled continuous descriptors for applicability domain
    scaled_desc_bbb = [bbb_df[c].iloc[0] for c in ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']]
    scaled_desc_pgp = [pgp_df[c].iloc[0] for c in [
        'MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds',
        'RadiusOfGyration', 'Asphericity', 'Eccentricity', 'SpherocityIndex', 'NPR1', 'NPR2'
    ]]

    domain_bbb = domain_estimator.evaluate_compound(mol, scaled_desc_bbb, "BBB", bbb_prob)
    domain_pgp = domain_estimator.evaluate_compound(mol, scaled_desc_pgp, "Pgp", pgp_prob)

    # Compute TreeSHAP Explanations
    bbb_expl = shap_explainer.explain_prediction(
        model_name="BBB",
        input_df=bbb_df,
        mol=mol,
        raw_descriptors=raw_bbb_desc,
        decision_threshold=0.5,
        top_n=5
    )

    pgp_expl = shap_explainer.explain_prediction(
        model_name="Pgp",
        input_df=pgp_df,
        mol=mol,
        raw_descriptors=raw_pgp_desc,
        decision_threshold=0.5,
        top_n=5
    )

    # SECTION 2: BBB Permeability Explainability
    print(f"{Colors.BOLD}{Colors.UNDERLINE}2. BLOOD-BRAIN BARRIER (BBB) PERMEABILITY EXPLAINABILITY (TreeSHAP){Colors.ENDC}")
    print(f"Predicted Class: {Colors.BOLD}{bbb_expl['class_label']}{Colors.ENDC} | Confidence: {bbb_prob*100:.1f}%")
    print(f"TreeSHAP Margin f(x): {bbb_expl['model_margin']:+.4f} (Base value phi_0 = {bbb_expl['base_value']:+.4f})")
    print(f"Biophysical Rationale: {bbb_expl['interpretation']['biophysical_basis']}")
    print(f"Reference Rule: {bbb_expl['interpretation']['rule_alignment']}")

    print(f"\n  {Colors.BOLD}Top Features Supporting Predicted BBB Outcome:{Colors.ENDC}")
    for item in bbb_expl['top_supporting']:
        val_str = f"{item['raw_value']:.2f}" if 'raw_value' in item else f"{item['input_value']:.2f}"
        subs = item.get('substructures', [])
        sub_str = f" | Substructure: {subs[0]['fragment_smiles']} ({subs[0]['functional_group']})" if subs else ""
        print(f"   * {item['feature']:<14} = {val_str:<8} -> Impact (phi) = {Colors.GREEN}{item['shap_value']:+.4f}{Colors.ENDC}{sub_str}")

    print(f"  {Colors.BOLD}Top Features Opposing Predicted BBB Outcome:{Colors.ENDC}")
    for item in bbb_expl['top_opposing']:
        val_str = f"{item['raw_value']:.2f}" if 'raw_value' in item else f"{item['input_value']:.2f}"
        subs = item.get('substructures', [])
        sub_str = f" | Substructure: {subs[0]['fragment_smiles']} ({subs[0]['functional_group']})" if subs else ""
        print(f"   * {item['feature']:<14} = {val_str:<8} -> Impact (phi) = {Colors.RED}{item['shap_value']:+.4f}{Colors.ENDC}{sub_str}")

    # SECTION 3: P-gp Efflux Pump Explainability
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}3. P-GLYCOPROTEIN (P-gp / ABCB1) EFFLUX EXPLAINABILITY (TreeSHAP & 3D Conformation){Colors.ENDC}")
    print(f"Predicted Class: {Colors.BOLD}{pgp_expl['class_label']}{Colors.ENDC} | Confidence: {pgp_prob*100:.1f}%")
    print(f"TreeSHAP Margin f(x): {pgp_expl['model_margin']:+.4f} (Base value phi_0 = {pgp_expl['base_value']:+.4f})")
    print(f"Efflux Cavity Mechanism: {pgp_expl['interpretation']['biophysical_basis']}")
    print(f"Reference Rule: {pgp_expl['interpretation']['rule_alignment']}")

    print(f"\n  {Colors.BOLD}Top Features Supporting Predicted P-gp Outcome:{Colors.ENDC}")
    for item in pgp_expl['top_supporting']:
        val_str = f"{item['raw_value']:.2f}" if 'raw_value' in item else f"{item['input_value']:.2f}"
        subs = item.get('substructures', [])
        sub_str = f" | Substructure: {subs[0]['fragment_smiles']} ({subs[0]['functional_group']})" if subs else ""
        print(f"   * {item['feature']:<18} = {val_str:<8} -> Impact (phi) = {Colors.GREEN}{item['shap_value']:+.4f}{Colors.ENDC}{sub_str}")

    print(f"  {Colors.BOLD}Top Features Opposing Predicted P-gp Outcome:{Colors.ENDC}")
    for item in pgp_expl['top_opposing']:
        val_str = f"{item['raw_value']:.2f}" if 'raw_value' in item else f"{item['input_value']:.2f}"
        subs = item.get('substructures', [])
        sub_str = f" | Substructure: {subs[0]['fragment_smiles']} ({subs[0]['functional_group']})" if subs else ""
        print(f"   * {item['feature']:<18} = {val_str:<8} -> Impact (phi) = {Colors.RED}{item['shap_value']:+.4f}{Colors.ENDC}{sub_str}")

    # SECTION 4: BBB - P-gp Shared Molecular Properties Analysis
    shared_prop_res = shared_analyzer.compare_compound_shared_properties(
        raw_descriptors=raw_pgp_desc,
        bbb_shap_result=bbb_expl,
        pgp_shap_result=pgp_expl,
        mol=mol
    )

    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}4. BBB - P-gp SHARED MOLECULAR PROPERTIES COMPARATIVE ANALYSIS{Colors.ENDC}")
    print(f"{'Property':<24} | {'Value':<8} | {'BBB Impact':<11} | {'P-gp Impact':<11} | {'Task Relationship':<24}")
    print("-" * 88)
    for row in shared_prop_res['continuous_table']:
        print(f"{row['property_name']:<24} | {row['molecular_value']:<8.2f} | {row['bbb_contribution']:<+11.4f} | {row['pgp_contribution']:<+11.4f} | {row['task_relationship']:<24}")
        print(f"  -> Interpretation: {row['scientific_interpretation']}")

    print(f"\n{Colors.BOLD}Cross-Task Synthesis:{Colors.ENDC} {shared_prop_res['task_divergence_summary']}")

    if shared_prop_res['shared_substructures']:
        print(f"\n{Colors.BOLD}Shared Active Pharmacophore Fragments (Morgan Bits in Both Models):{Colors.ENDC}")
        for sbit in shared_prop_res['shared_substructures']:
            print(f" * Bit {sbit['feature']}: Substructure '{sbit['fragment_smiles']}' ({sbit['functional_group']})")
            print(f"   Impacts: BBB phi = {sbit['bbb_shap']:+.4f} | P-gp phi = {sbit['pgp_shap']:+.4f} -> {sbit['direction_note']}")

    # SECTION 5: Applicability Domain & Uncertainty Assessment
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}5. APPLICABILITY DOMAIN & UNCERTAINTY QUANTIFICATION{Colors.ENDC}")
    print(f" * BBB Training Domain Status: {domain_bbb['domain_status']} (Max Tanimoto: {domain_bbb['max_tanimoto_similarity']:.4f}, Leverage: {domain_bbb['descriptor_leverage_distance']:.2f})")
    if domain_bbb['domain_warning']:
        print(f"   {Colors.YELLOW}[WARNING] {domain_bbb['domain_warning']}{Colors.ENDC}")
    print(f" * P-gp Training Domain Status: {domain_pgp['domain_status']} (Max Tanimoto: {domain_pgp['max_tanimoto_similarity']:.4f}, Leverage: {domain_pgp['descriptor_leverage_distance']:.2f})")
    if domain_pgp['domain_warning']:
        print(f"   {Colors.YELLOW}[WARNING] {domain_pgp['domain_warning']}{Colors.ENDC}")
    print(f" * Dual-Barrier Uncertainty Score: {max(domain_bbb['uncertainty_score'], domain_pgp['uncertainty_score']):.3f} (0 = absolute confidence, 1 = maximum ambiguity)")

    # SECTION 6: Counterfactuals
    counterfactuals = []
    if run_counterfactuals and counterfactual_engine is not None:
        print(f"\n{Colors.BOLD}{Colors.UNDERLINE}6. CHEMICALLY VALID COUNTERFACTUAL ANALOGS (MOLECULAR OPTIMIZATION){Colors.ENDC}")
        counterfactuals = counterfactual_engine.find_counterfactuals(
            smiles=smiles,
            current_bbb_pred=bbb_pred,
            current_pgp_pred=pgp_pred,
            target_endpoint="both",
            max_counterfactuals=3
        )
        if counterfactuals:
            for i, cf in enumerate(counterfactuals, 1):
                print(f"  [Analog {i}] Tanimoto: {cf['tanimoto_similarity']:.3f} | {cf['transformation']}")
                print(f"   SMILES: {cf['analog_smiles']}")
                print(f"   Outcome: {cf['objective_note']} (BBB Prob: {cf['bbb_prob']*100:.1f}%, P-gp Prob: {cf['pgp_prob']*100:.1f}%)")
        else:
            print("  No single-step transformation flipped the predicted classification within chemical sanity bounds.")

    print(f"\n{Colors.CYAN}{'='*85}{Colors.ENDC}\n")

    # Compile structured dictionary
    report_dict = {
        "smiles": smiles,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quadrant_category": quadrant_name,
        "quadrant_description": quadrant_desc,
        "bbb_pred": bbb_pred,
        "bbb_prob": float(bbb_prob),
        "bbb_class_label": bbb_expl['class_label'],
        "pgp_pred": pgp_pred,
        "pgp_prob": float(pgp_prob),
        "pgp_class_label": pgp_expl['class_label'],
        "raw_descriptors": raw_pgp_desc,
        "bbb_explanation": bbb_expl,
        "pgp_explanation": pgp_expl,
        "shared_properties": shared_prop_res,
        "applicability_domain_bbb": domain_bbb,
        "applicability_domain_pgp": domain_pgp,
        "counterfactuals": counterfactuals
    }

    return report_dict


def run_benchmark_test(
    bbb_model, pgp_model, bbb_means, bbb_stds, pgp_means, pgp_stds,
    shap_explainer, shared_analyzer, domain_estimator
):
    """Runs a validated test set of clinically characterized CNS and non-CNS reference molecules."""
    benchmarks = [
        ("Levetiracetam (Keppra)", "CC[C@@H](C(=O)N)N1CCCC1=O", "Refractory AED that evades P-gp efflux (Löscher p. 92)"),
        ("Loperamide (Imodium)", "CN(C)C(=O)CC(c1ccccc1)(c2ccccc2)CCN3CCC(CC3)(c4ccc(cc4)Cl)O", "Classic Paradoxical Drug (BBB+ but P-gp+ effluxed; Table 1)"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Classic CNS active stimulant (BBB+ / P-gp-)"),
        ("Diazepam (Valium)", "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13", "Lipophilic sedative; penetrates brain readily"),
        ("Penicillin G", "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C", "Peripheral antibiotic; poor BBB penetrant (BBB-)")
    ]

    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*85}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}     BENCHMARK VALIDATION SUITE: CLINICAL REFERENCE DRUGS (LÖSCHER 2005) {Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*85}{Colors.ENDC}\n")

    for drug_name, smiles, note in benchmarks:
        print(f"{Colors.CYAN}{Colors.BOLD}--- Testing Benchmark: {drug_name} ---{Colors.ENDC}")
        print(f"Pharmacological Grounding: {note}")
        print(f"SMILES: {smiles}")

        bbb_feats, raw_bbb, mol_bbb = extract_features_bbb(smiles)
        pgp_feats, raw_pgp, mol_pgp = extract_features_pgp(smiles)

        if bbb_feats is None or pgp_feats is None:
            print(f"{Colors.RED}Error extracting descriptors for {drug_name}{Colors.ENDC}\n")
            continue

        bbb_df = prepare_dataframe(bbb_feats[0], bbb_feats[1], bbb_means, bbb_stds, is_pgp=False)
        pgp_df = prepare_dataframe(pgp_feats[0], pgp_feats[1], pgp_means, pgp_stds, is_pgp=True)

        bbb_prob = float(bbb_model.predict_proba(bbb_df)[0][1])
        bbb_pred = int(bbb_prob > 0.5)

        pgp_prob = float(pgp_model.predict_proba(pgp_df)[0][1])
        pgp_pred = int(pgp_prob > 0.5)

        quadrant_name, quadrant_desc = determine_dual_barrier_quadrant(bbb_pred, pgp_pred)

        bbb_str = f"{Colors.GREEN}BBB+ ({bbb_prob*100:.1f}%){Colors.ENDC}" if bbb_pred == 1 else f"{Colors.RED}BBB- ({(1-bbb_prob)*100:.1f}%){Colors.ENDC}"
        pgp_str = f"{Colors.RED}Substrate ({pgp_prob*100:.1f}%){Colors.ENDC}" if pgp_pred == 1 else f"{Colors.GREEN}Non-Substrate ({(1-pgp_prob)*100:.1f}%){Colors.ENDC}"

        print(f"Predictions: BBB = {bbb_str} | P-gp = {pgp_str}")
        print(f"Classification: {Colors.BOLD}{quadrant_name}{Colors.ENDC}\n")


def main():
    parser = argparse.ArgumentParser(description="Dual-Barrier CNS Drug Predictor: BBB Permeability + P-gp Efflux XAI")
    parser.add_argument("--smiles", type=str, help="Single SMILES string to analyze")
    parser.add_argument("--explain", action="store_true", help="Display full biophysical and feature attribution explainability report")
    parser.add_argument("--counterfactual", action="store_true", help="Generate chemically valid counterfactual analog suggestions")
    parser.add_argument("--test-benchmark", action="store_true", help="Execute the clinical reference benchmark suite")
    parser.add_argument("--export-html", type=str, help="Save publication-grade scientific HTML report to specified path")
    parser.add_argument("--export-csv", type=str, help="Export prediction metrics and top attributions to CSV")
    parser.add_argument("--export-json", type=str, help="Save structured prediction and attribution dictionary to JSON")
    args = parser.parse_args()

    print(f"\n{Colors.CYAN}{Colors.BOLD}========================================================================{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}   Dual-Barrier CNS Drug Predictor (BBB + P-gp Efflux & Explainable AI)   {Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}========================================================================{Colors.ENDC}")

    # 1. Load BBB Model
    bbb_model_path = os.path.join(MODELS_DIR, "bbb_xgboost_optuna_best.json")
    if not os.path.exists(bbb_model_path):
        bbb_model_path = os.path.join(MODELS_DIR, "bbb_xgboost_model.json")

    bbb_model = xgb.XGBClassifier()
    bbb_model.load_model(bbb_model_path)

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
        print(f"{Colors.GREEN}SHAP Engine: Enabled (TreeSHAP active in margin space){Colors.ENDC}\n")
    else:
        print(f"{Colors.YELLOW}SHAP Engine: Using internal model attribution fallback{Colors.ENDC}\n")

    # Initialize Explainability Subsystems
    shap_explainer = DualBarrierShapExplainer(bbb_model, pgp_model)
    substructure_mapper = MorganSubstructureMapper()
    shared_analyzer = SharedPropertiesAnalyzer(substructure_mapper)
    domain_estimator = ApplicabilityDomainEstimator()
    counterfactual_engine = MolecularCounterfactualEngine(
        bbb_model, pgp_model, bbb_means, bbb_stds, pgp_means, pgp_stds
    )

    # If --test-benchmark is requested
    if args.test_benchmark:
        run_benchmark_test(
            bbb_model, pgp_model, bbb_means, bbb_stds, pgp_means, pgp_stds,
            shap_explainer, shared_analyzer, domain_estimator
        )
        return

    # Process CLI SMILES or Interactive Mode
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
                    run_benchmark_test(
                        bbb_model, pgp_model, bbb_means, bbb_stds, pgp_means, pgp_stds,
                        shap_explainer, shared_analyzer, domain_estimator
                    )
                    continue
                if not user_input:
                    continue
            else:
                user_input = candidate_list.pop(0)

            # Feature Extraction
            bbb_feats, raw_bbb_desc, mol = extract_features_bbb(user_input)
            pgp_feats, raw_pgp_desc, mol_3d = extract_features_pgp(user_input)

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

            # Model Predictions
            bbb_df = prepare_dataframe(bbb_fp, bbb_desc, bbb_means, bbb_stds, is_pgp=False)
            bbb_prob = float(bbb_model.predict_proba(bbb_df)[0][1])
            bbb_pred = int(bbb_prob > 0.5)

            pgp_df = prepare_dataframe(pgp_fp, pgp_desc, pgp_means, pgp_stds, is_pgp=True)
            pgp_prob = float(pgp_model.predict_proba(pgp_df)[0][1])
            pgp_pred = int(pgp_prob > 0.5)

            # Display Comprehensive Scientific Report
            report_data = display_full_scientific_report(
                smiles=user_input,
                mol=mol,
                raw_bbb_desc=raw_bbb_desc,
                raw_pgp_desc=raw_pgp_desc,
                bbb_df=bbb_df,
                pgp_df=pgp_df,
                bbb_model=bbb_model,
                pgp_model=pgp_model,
                bbb_pred=bbb_pred,
                bbb_prob=bbb_prob,
                pgp_pred=pgp_pred,
                pgp_prob=pgp_prob,
                shap_explainer=shap_explainer,
                shared_analyzer=shared_analyzer,
                domain_estimator=domain_estimator,
                counterfactual_engine=counterfactual_engine,
                run_counterfactuals=(args.counterfactual or interactive)
            )

            # Handle Export Requests
            if args.export_html:
                ScientificReportGenerator.export_html(report_data, args.export_html)
                print(f"{Colors.GREEN}[SAVED] Scientific HTML report exported to: {args.export_html}{Colors.ENDC}")

            if args.export_csv:
                ScientificReportGenerator.export_csv(report_data, args.export_csv)
                print(f"{Colors.GREEN}[SAVED] CSV summary exported to: {args.export_csv}{Colors.ENDC}")

            if args.export_json:
                ScientificReportGenerator.export_json(report_data, args.export_json)
                print(f"{Colors.GREEN}[SAVED] JSON data exported to: {args.export_json}{Colors.ENDC}")

            if not interactive:
                break

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Exiting...{Colors.ENDC}")
            break


if __name__ == "__main__":
    main()
