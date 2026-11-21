"""
Sequential Virtual Screen of the FDA-Approved Pharmacopeia (Novelty 2).

Applies the two-stage sequential cascade over the 2,405 launched small molecules
from the Broad Institute Drug Repurposing Hub:
  Stage 1: High-Throughput BBB Permeation Sieve (Fast 2D physicochemical + Morgan FP)
  Stage 2: Selective P-gp Efflux Gatekeeper (3D ETKDGv3 conformer geometry)

Crucial difference from Novelty 1:
In Novelty 1, all 2,405 drugs were forced through 3D conformer generation and parallel
dual-model inference simultaneously.
In Novelty 2, molecules that fail Stage 1 (BBB-) are immediately classified as
peripheral non-penetrants, completely bypassing Stage 2 conformer generation and inference.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.sequential_engine import SequentialCascadedEngine

DATA_PATH = os.path.join(BASE_DIR, "data", "repurposing", "fda_approved_repurposing_compounds.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_screen():
    print("=" * 80)
    print("NOVELTY 2: SEQUENTIAL CASCADED PHARMACOPEIA VIRTUAL SCREEN")
    print("=" * 80)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"FDA approved repurposing library not found at: {DATA_PATH}")

    df_drugs = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df_drugs)} FDA-approved drug candidates from {DATA_PATH}")

    engine = SequentialCascadedEngine()

    print("\nExecuting Stage-by-Stage Sequential Cascade...")
    start_time = time.perf_counter()

    records = []
    n_stage1_pass = 0
    n_stage1_reject = 0
    n_cns_leads = 0
    n_paradoxical = 0
    s1_times = []
    s2_times = []

    for idx, row in df_drugs.iterrows():
        smiles = row["canonical_smiles"] if "canonical_smiles" in row else row["smiles"]
        drug_name = row.get("pert_iname", row.get("name", f"Drug_{idx}"))
        disease_area = row.get("disease_area", "Unknown")
        indication = row.get("indication", "")

        res = engine.evaluate_molecule(smiles)
        s1_times.append(res.stage1_latency_ms)
        if res.stage2_evaluated:
            s2_times.append(res.stage2_latency_ms)

        if res.stage1_pass:
            n_stage1_pass += 1
            if res.final_disposition == "CNS_PENETRANT":
                n_cns_leads += 1
            elif res.final_disposition == "PARADOXICAL_EFFLUX":
                n_paradoxical += 1
        else:
            n_stage1_reject += 1

        records.append({
            "drug_name": drug_name,
            "smiles": smiles,
            "disease_area": disease_area,
            "indication": indication,
            "stage1_evaluated": res.stage1_evaluated,
            "stage1_pass": res.stage1_pass,
            "prob_bbb_permeant": round(res.stage1_prob_bbb, 4),
            "stage2_evaluated": res.stage2_evaluated,
            "stage2_pass": res.stage2_pass,
            "prob_pgp_efflux": round(res.stage2_prob_pgp, 4) if res.stage2_prob_pgp is not None else None,
            "final_disposition": res.final_disposition,
            "composite_cns_score": round(res.composite_cns_score, 4),
            "compute_saved": res.compute_saved,
            "stage1_latency_ms": round(res.stage1_latency_ms, 2),
            "stage2_latency_ms": round(res.stage2_latency_ms, 2),
            "total_latency_ms": round(res.total_latency_ms, 2)
        })

    elapsed_total = time.perf_counter() - start_time
    df_out = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS_DIR, "sequential_repurposing_screen.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved screen results to: {out_csv}")

    # Summary statistics
    n_total = len(df_drugs)
    stage1_drop_pct = (n_stage1_reject / n_total) * 100.0
    stage2_eval_pct = (n_stage1_pass / n_total) * 100.0
    cns_yield_pct = (n_cns_leads / n_total) * 100.0
    paradox_pct = (n_paradoxical / n_stage1_pass) * 100.0 if n_stage1_pass > 0 else 0.0

    print("\n" + "=" * 80)
    print("SEQUENTIAL CASCADED SIEVE ATTRITION SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated Drugs (Stage 1 Input) : {n_total}")
    print(f"Stage 1 BBB Sieve Passed (Permeant)   : {n_stage1_pass} ({stage2_eval_pct:.1f}%) -> Advanced to Stage 2")
    print(f"Stage 1 BBB Sieve Rejected (Peripheral): {n_stage1_reject} ({stage1_drop_pct:.1f}%) -> Early Terminated")
    print(f"Stage 2 P-gp Gate Passed (CNS Leads)  : {n_cns_leads} ({cns_yield_pct:.1f}% overall yield)")
    print(f"Stage 2 P-gp Gate Caught (Paradoxical): {n_paradoxical} ({paradox_pct:.1f}% of permeants)")
    print("-" * 80)
    print(f"Early Compute Savings Ratio           : {stage1_drop_pct:.1f}% of Stage 2 3D conformer work saved!")
    print(f"Mean Stage 1 Latency (2D Sieve)       : {np.mean(s1_times):.2f} ms / molecule")
    if s2_times:
        print(f"Mean Stage 2 Latency (3D Gatekeeper)  : {np.mean(s2_times):.2f} ms / molecule")
    print(f"Total Sequential Screen Runtime       : {elapsed_total:.2f} s")
    print("=" * 80)

    # Save summary report
    summary_txt = os.path.join(RESULTS_DIR, "sequential_screen_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"NOVELTY 2 SEQUENTIAL SCREEN SUMMARY\n")
        f.write(f"Total Candidates: {n_total}\n")
        f.write(f"Stage 1 Passed: {n_stage1_pass} ({stage2_eval_pct:.1f}%)\n")
        f.write(f"Stage 1 Rejected (Early Terminated): {n_stage1_reject} ({stage1_drop_pct:.1f}%)\n")
        f.write(f"Stage 2 CNS Leads: {n_cns_leads} ({cns_yield_pct:.1f}%)\n")
        f.write(f"Stage 2 Paradoxical Efflux: {n_paradoxical} ({paradox_pct:.1f}%)\n")
        f.write(f"Early Compute Saved: {stage1_drop_pct:.1f}%\n")
        f.write(f"Total Elapsed: {elapsed_total:.2f} s\n")

    return df_out


if __name__ == "__main__":
    run_screen()
