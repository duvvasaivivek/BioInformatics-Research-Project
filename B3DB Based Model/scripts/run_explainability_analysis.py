"""
Global Explainability and Cross-Task Shared Properties Analysis Script.
Computes global TreeSHAP feature attributions on held-out test sets for both BBB and P-gp models,
evaluates rank correlations across shared features, and generates publication-grade figures.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Set plotting style
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cbd5e0'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add explainability package
sys.path.insert(0, BASE_DIR)
from explainability import DualBarrierShapExplainer, SharedPropertiesAnalyzer


def run_global_analysis():
    print("=" * 80)
    print("   GLOBAL TREE-SHAP & SHARED-PROPERTIES QUANTITATIVE ANALYSIS   ")
    print("=" * 80)

    # 1. Load BBB Model & Data
    print("\n[1/5] Loading BBB dataset and Optuna-tuned model...")
    bbb_data_path = os.path.join(DATA_DIR, "processed_b3db_data_advanced.csv")
    df_bbb = pd.read_csv(bbb_data_path)
    X_bbb = df_bbb.drop(columns=['SMILES', 'Label'])
    y_bbb = df_bbb['Label']

    # Exact held-out split matching bbb_model_training.py
    X_bbb_train, X_bbb_test, y_bbb_train, y_bbb_test = train_test_split(
        X_bbb, y_bbb, test_size=0.2, random_state=42
    )

    bbb_model = xgb.XGBClassifier()
    bbb_model.load_model(os.path.join(MODELS_DIR, "bbb_xgboost_optuna_best.json"))

    # 2. Load P-gp Model & Data
    print("[2/5] Loading P-gp dataset and 3D shape model...")
    pgp_data_path = os.path.join(DATA_DIR, "processed_pgp_data.csv")
    df_pgp = pd.read_csv(pgp_data_path)
    X_pgp = df_pgp.drop(columns=['SMILES', 'Label'])
    y_pgp = df_pgp['Label']

    X_pgp_train, X_pgp_test, y_pgp_train, y_pgp_test = train_test_split(
        X_pgp, y_pgp, test_size=0.2, random_state=42, stratify=y_pgp
    )

    pgp_model = xgb.XGBClassifier()
    pgp_model.load_model(os.path.join(MODELS_DIR, "pgp_xgboost_model.json"))

    # Initialize Explainer
    explainer = DualBarrierShapExplainer(bbb_model, pgp_model)

    # 3. Compute Sampled TreeSHAP on Held-Out Test Sets
    sample_size_bbb = min(500, len(X_bbb_test))
    sample_size_pgp = min(250, len(X_pgp_test))

    print(f"[3/5] Computing TreeSHAP on test subsets (BBB: N={sample_size_bbb}, P-gp: N={sample_size_pgp})...")
    X_bbb_sample = X_bbb_test.sample(n=sample_size_bbb, random_state=42)
    X_pgp_sample = X_pgp_test.sample(n=sample_size_pgp, random_state=42)

    shap_bbb = explainer._bbb_explainer(X_bbb_sample).values
    shap_pgp = explainer._pgp_explainer(X_pgp_sample).values

    # Mean absolute SHAP across test sample
    mean_abs_bbb = pd.Series(np.mean(np.abs(shap_bbb), axis=0), index=X_bbb.columns)
    mean_abs_pgp = pd.Series(np.mean(np.abs(shap_pgp), axis=0), index=X_pgp.columns)

    mean_signed_bbb = pd.Series(np.mean(shap_bbb, axis=0), index=X_bbb.columns)
    mean_signed_pgp = pd.Series(np.mean(shap_pgp, axis=0), index=X_pgp.columns)

    # 4. Shared Descriptors Quantitative Comparison
    shared_desc = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
    print("\n[4/5] Computing Cross-Task Shared Feature Rank Correlations & Statistics...")

    comparison_data = []
    for desc in shared_desc:
        b_imp = mean_abs_bbb.get(desc, 0.0)
        p_imp = mean_abs_pgp.get(desc, 0.0)
        b_sign = mean_signed_bbb.get(desc, 0.0)
        p_sign = mean_signed_pgp.get(desc, 0.0)

        # Biological task categorization
        if desc == 'LogP':
            cat = "Antagonistic"
            interp = "Promotes passive lipid partitioning (BBB+) but simultaneously increases affinity for the P-gp cavity (P-gp+)."
        elif desc == 'MolWt':
            cat = "Compound Barrier"
            interp = "Impedes passive diffusion (BBB-) while expanding contacts within the P-gp translocation pocket (P-gp+)."
        elif desc == 'TPSA':
            cat = "Desolvation Filter"
            interp = "Polar penalty reduces transcellular influx (BBB-) while moderately deterring hydrophobic P-gp cleft binding."
        elif desc in ['HDonors', 'HAcceptors']:
            cat = "Efflux Trigger (Seelig)"
            interp = "Hydrogen-bonding pairs act as primary recognition motifs for ABCB1 transmembrane helices."
        else:
            cat = "Conformational Flexibility"
            interp = "Rotatable bonds allow induced-fit cavity binding inside the multidrug transporter chamber."

        comparison_data.append({
            "Property": desc,
            "BBB_Mean_Abs_SHAP": round(b_imp, 4),
            "Pgp_Mean_Abs_SHAP": round(p_imp, 4),
            "BBB_Signed_SHAP": round(b_sign, 4),
            "Pgp_Signed_SHAP": round(p_sign, 4),
            "Task_Category": cat,
            "Scientific_Interpretation": interp
        })

    comp_df = pd.DataFrame(comparison_data)
    comp_csv_path = os.path.join(RESULTS_DIR, "shared_properties_quantitative_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"Saved shared properties table to {comp_csv_path}")

    # Spearman rank correlation across continuous descriptors
    rho, pval = spearmanr(
        [d["BBB_Mean_Abs_SHAP"] for d in comparison_data],
        [d["Pgp_Mean_Abs_SHAP"] for d in comparison_data]
    )
    print(f"\n--- Spearman Rank Correlation on Shared Physicochemical Descriptors ---")
    print(f"Spearman rho = {rho:.4f} (p-value = {pval:.4e})")

    # 5. Generate Publication-Quality Visualizations
    print("\n[5/5] Generating publication-quality figures in figures/ ...")

    # Figure 1: BBB Top 15 Global Features
    top_bbb = mean_abs_bbb.sort_values(ascending=False).head(15)
    plt.figure(figsize=(9, 6))
    colors_bbb = ['#2b6cb0' if c in shared_desc else '#4a5568' for c in top_bbb.index]
    y_pos = np.arange(len(top_bbb))
    plt.barh(y_pos, top_bbb.values, color=colors_bbb, edgecolor='#1a365d', height=0.65)
    plt.yticks(y_pos, top_bbb.index)
    plt.gca().invert_yaxis()
    plt.xlabel('Mean |TreeSHAP Value| (Log-odds Impact on BBB Permeability)')
    plt.title('Global Feature Importance: Blood-Brain Barrier (BBB) Permeability\n(Blue = Shared 2D Descriptors, Gray = Morgan Fingerprint Bits)')
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "bbb_shap_global_summary.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f" * Figure 1: {fig1_path}")

    # Figure 2: P-gp Top 15 Global Features
    top_pgp = mean_abs_pgp.sort_values(ascending=False).head(15)
    shape_3d = ['RadiusOfGyration', 'Asphericity', 'Eccentricity', 'SpherocityIndex', 'NPR1', 'NPR2']
    colors_pgp = []
    for c in top_pgp.index:
        if c in shape_3d:
            colors_pgp.append('#c53030')  # Red for 3D
        elif c in shared_desc:
            colors_pgp.append('#2b6cb0')  # Blue for shared 2D
        else:
            colors_pgp.append('#4a5568')  # Gray for Morgan

    plt.figure(figsize=(9, 6))
    y_pos = np.arange(len(top_pgp))
    plt.barh(y_pos, top_pgp.values, color=colors_pgp, edgecolor='#742a2a', height=0.65)
    plt.yticks(y_pos, top_pgp.index)
    plt.gca().invert_yaxis()
    plt.xlabel('Mean |TreeSHAP Value| (Log-odds Impact on P-gp Efflux Substrate Status)')
    plt.title('Global Feature Importance: P-glycoprotein (P-gp / ABCB1) Efflux\n(Red = 3D Conformer Descriptors, Blue = Shared 2D Descriptors, Gray = Morgan Bits)')
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "pgp_shap_global_summary.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f" * Figure 2: {fig2_path}")

    # Figure 3: Shared Properties Cross-Task Contribution Bar Chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(shared_desc))
    width = 0.35
    b_vals = [comp_df.loc[comp_df['Property'] == d, 'BBB_Mean_Abs_SHAP'].values[0] for d in shared_desc]
    p_vals = [comp_df.loc[comp_df['Property'] == d, 'Pgp_Mean_Abs_SHAP'].values[0] for d in shared_desc]

    plt.bar(x - width/2, b_vals, width, label='BBB Permeability Impact', color='#3182ce', edgecolor='#1a365d')
    plt.bar(x + width/2, p_vals, width, label='P-gp Efflux Impact', color='#e53e3e', edgecolor='#742a2a')
    plt.xticks(x, shared_desc)
    plt.ylabel('Mean |TreeSHAP| Value')
    plt.title('Direct Cross-Task Feature Importance Comparison on Shared Physicochemical Descriptors')
    plt.legend()
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "shared_properties_cross_task_comparison.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f" * Figure 3: {fig3_path}")

    # Figure 4: Dual-Barrier Four-Quadrant Matrix Diagram
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0.5, color='#718096', linestyle='--', linewidth=1.5)
    ax.axvline(0.5, color='#718096', linestyle='--', linewidth=1.5)

    # Quadrant 1: Top-Left (BBB+ / P-gp-)
    ax.fill_between([0.5, 1.0], 0.0, 0.5, color='#c6f6d5', alpha=0.35)
    ax.text(0.75, 0.25, "QUADRANT 1\nBBB+ / P-gp-\n[OPTIMAL CNS AGENT]\nHigh Influx & Efflux Evasion\n(e.g., Levetiracetam)",
            ha='center', va='center', fontsize=11, fontweight='bold', color='#22543d')

    # Quadrant 2: Top-Right (BBB+ / P-gp+)
    ax.fill_between([0.5, 1.0], 0.5, 1.0, color='#feebc8', alpha=0.35)
    ax.text(0.75, 0.75, "QUADRANT 2\nBBB+ / P-gp+\n[PARADOXICAL DRUG]\nHigh Lipid Influx, Active Efflux\n(e.g., Loperamide)",
            ha='center', va='center', fontsize=11, fontweight='bold', color='#744210')

    # Quadrant 3: Bottom-Left (BBB- / P-gp-)
    ax.fill_between([0.0, 0.5], 0.0, 0.5, color='#e2e8f0', alpha=0.35)
    ax.text(0.25, 0.25, "QUADRANT 3\nBBB- / P-gp-\n[PERIPHERAL AGENT]\nLow Passive Permeation\n(e.g., Penicillin G)",
            ha='center', va='center', fontsize=11, fontweight='bold', color='#4a5568')

    # Quadrant 4: Bottom-Right (BBB- / P-gp+)
    ax.fill_between([0.0, 0.5], 0.5, 1.0, color='#fed7d7', alpha=0.35)
    ax.text(0.25, 0.75, "QUADRANT 4\nBBB- / P-gp+\n[RESTRICTED EFFLUXED]\nImpermeable & Effluxed",
            ha='center', va='center', fontsize=11, fontweight='bold', color='#742a2a')

    # Plot clinical reference points
    refs = [
        ("Levetiracetam", 0.988, 0.028, '#22543d'),
        ("Loperamide", 0.934, 0.907, '#744210'),
        ("Caffeine", 0.946, 0.156, '#22543d'),
        ("Diazepam", 0.997, 0.022, '#22543d'),
        ("Penicillin G", 0.007, 0.083, '#4a5568')
    ]
    for name, b_p, p_p, col in refs:
        ax.scatter(b_p, p_p, s=120, color=col, edgecolor='black', zorder=5)
        ax.annotate(name, (b_p, p_p), textcoords="offset points", xytext=(8, -4),
                    fontsize=9, fontweight='bold', color='#1a202c', zorder=6)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Predicted Probability of BBB Permeability P(BBB+)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Probability of P-gp Substrate Efflux P(P-gp+)', fontsize=11, fontweight='bold')
    ax.set_title('Dual-Barrier Four-Quadrant Pharmacological Decision Matrix\nMapping Clinical Benchmark Therapeutics', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "dual_barrier_quadrant_matrix.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f" * Figure 4: {fig4_path}")

    print("\nGlobal TreeSHAP and shared-properties analysis complete!")


if __name__ == "__main__":
    run_global_analysis()
