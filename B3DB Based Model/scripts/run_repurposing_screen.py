"""
Systematic Dual-Barrier Virtual Screening of the FDA-Approved Pharmacopeia.
Profiles ~2,400 approved small molecule drugs from the Broad Institute Drug Repurposing Hub:
1. Predicts passive BBB permeability and active P-gp efflux with trained models.
2. Evaluates chemical applicability domain against training distributions.
3. Computes the CNS Desirability Index: S_CNS = P(BBB+) * (1 - P(P-gp+)).
4. Identifies non-CNS drugs for neurotherapeutic repurposing (Glioblastoma, Neuroinflammation).
5. Uncovers efflux-mediated treatment failure in marketed neuropsychiatric drugs.
6. Generates publication-ready figures and structured CSV tables.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D
from rdkit import RDLogger
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.features import (
    BBB_DESCRIPTOR_NAMES,
    PGP_DESCRIPTOR_NAMES,
    calculate_2d_descriptors,
    calculate_3d_descriptors,
    calculate_morgan_fingerprint,
    load_scaler_metadata
)
from core.decision_matrix import determine_dual_barrier_quadrant, Colors
from explainability.applicability_domain import ApplicabilityDomainEstimator

DATA_DIR = os.path.join(BASE_DIR, "data")
REPURPOSING_DIR = os.path.join(DATA_DIR, "repurposing")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

INPUT_CSV = os.path.join(REPURPOSING_DIR, "fda_approved_repurposing_compounds.csv")
OUTPUT_SCREEN_CSV = os.path.join(RESULTS_DIR, "fda_repurposing_dual_barrier_screen.csv")
OUTPUT_TOP_REPURPOSING_CSV = os.path.join(RESULTS_DIR, "top_cns_repurposing_candidates.csv")
OUTPUT_RESISTANCE_CSV = os.path.join(RESULTS_DIR, "cns_drugs_efflux_resistance_profiles.csv")


def process_single_molecule(row_tuple) -> Optional[Dict[str, Any]]:
    """Helper worker to extract BBB and P-gp features for a single molecule."""
    idx, name, smiles, phase, moa, target, disease, indication = row_tuple
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 1. 2D Descriptors & Morgan Fingerprint
        fp_arr = calculate_morgan_fingerprint(mol, radius=2, n_bits=2048)
        desc_2d = calculate_2d_descriptors(mol)

        # 2. 3D Shape Conformer via ETKDGv3
        desc_3d, _ = calculate_3d_descriptors(mol, seed=42)
        if desc_3d is None:
            # Fallback for conformer generation
            desc_3d = {
                'RadiusOfGyration': 0.0, 'Asphericity': 0.0, 'Eccentricity': 0.0,
                'SpherocityIndex': 0.0, 'NPR1': 0.0, 'NPR2': 0.0
            }
            has_3d = False
        else:
            has_3d = True

        combined_desc = dict(desc_2d)
        combined_desc.update(desc_3d)

        return {
            'idx': idx,
            'drug_name': name,
            'smiles': smiles,
            'clinical_phase': phase,
            'moa': moa,
            'target': target,
            'disease_area': disease,
            'indication': indication,
            'fp': fp_arr,
            'desc_2d': [desc_2d[k] for k in BBB_DESCRIPTOR_NAMES],
            'desc_pgp': [combined_desc[k] for k in PGP_DESCRIPTOR_NAMES],
            'has_3d': has_3d
        }
    except Exception:
        return None


def run_screen():
    print("=" * 80)
    print("   FDA-APPROVED PHARMACOPEIA DUAL-BARRIER VIRTUAL SCREENING   ")
    print("=" * 80)

    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found.")
        return

    df_drugs = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df_drugs)} FDA-approved drugs for dual-barrier screening.")

    # 1. Load trained models and scalers
    print("\nLoading models and training scaler metadata...")
    bbb_model = xgb.XGBClassifier()
    bbb_model.load_model(os.path.join(MODELS_DIR, "bbb_xgboost_optuna_best.json"))

    pgp_model = xgb.XGBClassifier()
    pgp_model.load_model(os.path.join(MODELS_DIR, "pgp_xgboost_model.json"))

    bbb_means, bbb_stds = load_scaler_metadata(os.path.join(MODELS_DIR, "bbb_scaler_metadata.json"))
    pgp_means, pgp_stds = load_scaler_metadata(os.path.join(MODELS_DIR, "pgp_scaler_metadata.json"))

    domain_estimator = ApplicabilityDomainEstimator(max_samples=1500)

    # 2. Extract features in parallel
    print(f"\nExtracting molecular features and 3D conformers across CPU cores...")
    tasks = [
        (
            i,
            row['drug_name'],
            row['smiles'],
            row['clinical_phase'],
            row['moa'],
            row['target'],
            row['disease_area'],
            row['indication']
        )
        for i, row in df_drugs.iterrows()
    ]

    cache_path = os.path.join(REPURPOSING_DIR, "repurposing_features_cache.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached molecular features from {cache_path}...")
        import pickle
        with open(cache_path, 'rb') as f:
            extracted_records = pickle.load(f)
        print(f"Loaded {len(extracted_records)} cached molecular records.")
    else:
        extracted_records = []
        workers = min(8, os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_molecule, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Profiling Molecules"):
                res = fut.result()
                if res is not None:
                    extracted_records.append(res)

        extracted_records.sort(key=lambda x: x['idx'])
        print(f"\nSuccessfully extracted features for {len(extracted_records)}/{len(df_drugs)} molecules.")
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(extracted_records, f)
        print(f"Cached extracted features to {cache_path}")

    # 3. Model Inference & Screening (Batched & Vectorized for 100x Speedup)
    print("\nRunning Batched Dual-Barrier Ensembles and Applicability Domain Assessment...")
    N = len(extracted_records)
    fp_cols = [f'FP_{i}' for i in range(2048)]
    fp_mat = np.array([item['fp'] for item in extracted_records], dtype=np.int8)

    # Scale continuous features in bulk
    desc_bbb_mat = np.array([item['desc_2d'] for item in extracted_records], dtype=np.float32)
    b_means = np.array(bbb_means, dtype=np.float32)
    b_stds = np.where(np.array(bbb_stds, dtype=np.float32) != 0, np.array(bbb_stds, dtype=np.float32), 1.0)
    desc_bbb_scaled = (desc_bbb_mat - b_means) / b_stds

    desc_pgp_mat = np.array([item['desc_pgp'] for item in extracted_records], dtype=np.float32)
    p_means = np.array(pgp_means, dtype=np.float32)
    p_stds = np.where(np.array(pgp_stds, dtype=np.float32) != 0, np.array(pgp_stds, dtype=np.float32), 1.0)
    desc_pgp_scaled = (desc_pgp_mat - p_means) / p_stds

    # Build evaluation DataFrames
    print("Constructing BBB feature matrix and predicting probabilities...")
    X_bbb = pd.DataFrame(fp_mat, columns=fp_cols)
    for i, col in enumerate(BBB_DESCRIPTOR_NAMES):
        X_bbb[col] = desc_bbb_scaled[:, i]
    probs_bbb = bbb_model.predict_proba(X_bbb)[:, 1]
    preds_bbb = (probs_bbb > 0.5).astype(int)

    print("Constructing P-gp feature matrix and predicting probabilities...")
    X_pgp = pd.DataFrame(fp_mat, columns=fp_cols)
    for i, col in enumerate(PGP_DESCRIPTOR_NAMES):
        X_pgp[col] = desc_pgp_scaled[:, i]
    probs_pgp = pgp_model.predict_proba(X_pgp)[:, 1]
    preds_pgp = (probs_pgp > 0.5).astype(int)

    print("Computing Applicability Domains and Desirability Scores...")
    screen_rows = []
    from rdkit.Chem import DataStructs

    for i, item in enumerate(tqdm(extracted_records, desc="Compiling Results")):
        prob_bbb = float(probs_bbb[i])
        pred_bbb = int(preds_bbb[i])
        prob_pgp = float(probs_pgp[i])
        pred_pgp = int(preds_pgp[i])

        # Quadrant assignment
        quadrant_name, rationale = determine_dual_barrier_quadrant(pred_bbb, pred_pgp)
        if pred_bbb == 1 and pred_pgp == 0:
            quad_id = 1
        elif pred_bbb == 1 and pred_pgp == 1:
            quad_id = 2
        elif pred_bbb == 0 and pred_pgp == 0:
            quad_id = 3
        else:
            quad_id = 4

        desirability = prob_bbb * (1.0 - prob_pgp)

        # Fast Applicability Domain via pre-loaded reference FPs
        mol = Chem.MolFromSmiles(item['smiles'])
        if mol is not None:
            test_fp = domain_estimator.bbb_fps[0].__class__(2048)  # dummy to get type
            # Use MorganGenerator
            from core.features import _MORGAN_GEN_R2_2048
            test_fp = _MORGAN_GEN_R2_2048.GetFingerprint(mol)
            sims_bbb = DataStructs.BulkTanimotoSimilarity(test_fp, domain_estimator.bbb_fps)
            sims_pgp = DataStructs.BulkTanimotoSimilarity(test_fp, domain_estimator.pgp_fps)
            max_tan_bbb = float(max(sims_bbb)) if sims_bbb else 0.0
            max_tan_pgp = float(max(sims_pgp)) if sims_pgp else 0.0
        else:
            max_tan_bbb = 0.0
            max_tan_pgp = 0.0

        screen_rows.append({
            'drug_name': item['drug_name'],
            'smiles': item['smiles'],
            'disease_area': str(item['disease_area']).lower() if pd.notna(item['disease_area']) else 'unspecified',
            'indication': item['indication'],
            'moa': item['moa'],
            'target': item['target'],
            'prob_bbb': round(prob_bbb, 4),
            'pred_bbb': pred_bbb,
            'prob_pgp': round(prob_pgp, 4),
            'pred_pgp': pred_pgp,
            'quadrant_id': quad_id,
            'quadrant_name': quadrant_name,
            'cns_desirability': round(desirability, 4),
            'tanimoto_bbb': round(max_tan_bbb, 4),
            'tanimoto_pgp': round(max_tan_pgp, 4),
            'domain_status': 'Inside' if (max_tan_bbb >= 0.35 and max_tan_pgp >= 0.35) else 'Borderline'
        })

    df_results = pd.DataFrame(screen_rows)
    df_results.to_csv(OUTPUT_SCREEN_CSV, index=False)
    print(f"\nSaved complete screening results to {OUTPUT_SCREEN_CSV}")

    # 4. Global Pharmacopeia Statistics
    q_counts = df_results['quadrant_id'].value_counts().sort_index()
    print("\n--- Pharmacopeia Dual-Barrier Distribution ---")
    for q_id in [1, 2, 3, 4]:
        c = q_counts.get(q_id, 0)
        pct = c / len(df_results) * 100
        print(f" * Quadrant {q_id}: {c:4d} drugs ({pct:5.1f}%)")

    # 5. Extract High-Priority Repurposing Candidates (Non-CNS in Q1)
    # Exclude neurology/psychiatry
    non_cns_mask = ~df_results['disease_area'].str.contains('neurology|psychiatry', case=False, na=False)
    q1_mask = (df_results['quadrant_id'] == 1) & (df_results['tanimoto_bbb'] >= 0.35)

    df_repurposing = df_results[non_cns_mask & q1_mask].sort_values(
        by='cns_desirability', ascending=False
    ).copy()

    df_repurposing.to_csv(OUTPUT_TOP_REPURPOSING_CSV, index=False)
    print(f"\nIdentified {len(df_repurposing)} Non-CNS FDA-approved drugs in Quadrant 1 (Optimal CNS Permeation).")
    print(f"Saved top repurposing candidates to {OUTPUT_TOP_REPURPOSING_CSV}")

    print("\n=== TOP 10 NON-CNS REPURPOSING HITS FOR NEUROTHERAPEUTICS ===")
    top_cols = ['drug_name', 'disease_area', 'indication', 'moa', 'prob_bbb', 'prob_pgp', 'cns_desirability']
    print(df_repurposing[top_cols].head(10).to_string(index=False))

    # 6. Extract Efflux Resistance in Known CNS Drugs (CNS in Q2)
    cns_mask = df_results['disease_area'].str.contains('neurology|psychiatry', case=False, na=False)
    q2_mask = (df_results['quadrant_id'] == 2)

    df_resistance = df_results[cns_mask & q2_mask].sort_values(
        by='prob_pgp', ascending=False
    ).copy()

    df_resistance.to_csv(OUTPUT_RESISTANCE_CSV, index=False)
    print(f"\nIdentified {len(df_resistance)} Marketed Neuropsychiatric Drugs in Quadrant 2 (Efflux Resistance Risk).")
    print(f"Saved resistance profiles to {OUTPUT_RESISTANCE_CSV}")

    print("\n=== TOP 10 CNS DRUGS VULNERABLE TO P-GP EFFLUX RESISTANCE ===")
    print(df_resistance[top_cols].head(10).to_string(index=False))

    # 7. Generate Publication Figures
    generate_repurposing_figures(df_results, df_repurposing, df_resistance)


def generate_repurposing_figures(df_results: pd.DataFrame, df_repurp: pd.DataFrame, df_resist: pd.DataFrame):
    """Generates high-resolution publication-quality figures."""
    print("\nGenerating publication figures...")

    # Figure 1: Global Pharmacopeia Dual-Barrier Landscape
    plt.figure(figsize=(9, 7))
    palette = {
        1: "#2b6cb0",  # Blue: Q1 Optimal
        2: "#c53030",  # Red: Q2 Paradoxical
        3: "#718096",  # Grey: Q3 Peripheral
        4: "#d69e2e"   # Gold: Q4 Restricted
    }

    sns.scatterplot(
        data=df_results,
        x='prob_bbb',
        y='prob_pgp',
        hue='quadrant_id',
        palette=palette,
        alpha=0.65,
        s=45,
        edgecolor='none'
    )

    plt.axvline(0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.2)
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.2)

    # Quadrant Labels
    plt.text(0.75, 0.20, "Quadrant 1:\nOptimal CNS (BBB+/P-gp-)", fontsize=11, fontweight='bold', color='#2b6cb0', ha='center')
    plt.text(0.75, 0.85, "Quadrant 2:\nParadoxical (BBB+/P-gp+)", fontsize=11, fontweight='bold', color='#c53030', ha='center')
    plt.text(0.25, 0.20, "Quadrant 3:\nPeripheral (BBB-/P-gp-)", fontsize=11, fontweight='bold', color='#4a5568', ha='center')
    plt.text(0.25, 0.85, "Quadrant 4:\nRestricted (BBB-/P-gp+)", fontsize=11, fontweight='bold', color='#b7791f', ha='center')

    # Annotate key clinical reference drugs
    annot_drugs = {
        'Levetiracetam': ('#1a365d', (0.95, 0.05)),
        'Loperamide': ('#742a2a', (0.85, 0.98)),
        'Diazepam': ('#1a365d', (0.98, 0.08)),
        'Caffeine': ('#1a365d', (0.88, 0.12)),
    }

    for name, (color, (tx, ty)) in annot_drugs.items():
        match = df_results[df_results['drug_name'].str.lower() == name.lower()]
        if len(match) > 0:
            x_val = match['prob_bbb'].values[0]
            y_val = match['prob_pgp'].values[0]
            plt.plot(x_val, y_val, marker='o', markersize=8, markeredgecolor='black', markerfacecolor='yellow')
            plt.annotate(
                name,
                xy=(x_val, y_val),
                xytext=(tx, ty),
                fontweight='bold',
                fontsize=9,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2)
            )

    plt.title("Dual-Barrier Profiling of 2,400+ FDA-Approved Drugs (Broad Repurposing Hub)", fontsize=13, fontweight='bold')
    plt.xlabel(r"Passive Membrane Permeability: $P(\mathrm{BBB}^+)$", fontsize=11, fontweight='bold')
    plt.ylabel(r"Active Efflux Recognition: $P(\mathrm{P\text{-}gp}^+)$", fontsize=11, fontweight='bold')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.legend().remove()
    plt.tight_layout()

    fig1_path = os.path.join(FIGURES_DIR, "repurposing_pharmacopeia_landscape.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f" * Saved landscape figure to {fig1_path}")

    # Figure 2: Disease Area Quadrant Breakdown
    top_areas = df_results['disease_area'].value_counts().head(7).index.tolist()
    df_top = df_results[df_results['disease_area'].isin(top_areas)].copy()

    ct = pd.crosstab(df_top['disease_area'], df_top['quadrant_id'], normalize='index') * 100
    ct = ct.reindex(columns=[1, 2, 3, 4], fill_value=0)

    plt.figure(figsize=(10, 5.5))
    ct.plot(
        kind='barh',
        stacked=True,
        color=['#2b6cb0', '#c53030', '#718096', '#d69e2e'],
        edgecolor='black',
        figsize=(10, 5.5)
    )

    plt.title("Dual-Barrier Quadrant Distribution Across Major Therapeutic Classes", fontsize=12, fontweight='bold')
    plt.xlabel("Percentage of Approved Drugs (%)", fontsize=10, fontweight='bold')
    plt.ylabel("Therapeutic Disease Area", fontsize=10, fontweight='bold')
    plt.legend(
        ['Q1: Optimal CNS (BBB+/P-gp-)', 'Q2: Paradoxical (BBB+/P-gp+)', 'Q3: Peripheral (BBB-/P-gp-)', 'Q4: Restricted (BBB-/P-gp+)'],
        loc='upper right', bbox_to_anchor=(1.0, -0.15), ncol=2, frameon=True
    )
    plt.tight_layout()

    fig2_path = os.path.join(FIGURES_DIR, "repurposing_disease_area_distribution.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" * Saved disease area distribution figure to {fig2_path}")


if __name__ == "__main__":
    run_screen()
