"""
Comprehensive Verification Suite for Novelty 2 (Sequential Cascaded Sieve & Transfer).

Guarantees all invariants, contracts, and regression thresholds for the sequential architecture:
  1. Directory & Artifact Integrity
  2. Scaler & Model Schema Contracts
  3. Sequential Sieve Gating Invariants (Stage 1 drop -> Stage 2 bypass)
  4. Clinical Benchmark Behavioral Contracts (Levetiracetam, Loperamide, Penicillin G)
  5. Sequential Transfer Learning GNN Invariants (Encoder trunk extraction & parameter shapes)
  6. Early Termination Computation Savings Validation

Exit code is 0 only if all checks pass.
"""

import os
import sys
from typing import List, Tuple
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.sequential_engine import SequentialCascadedEngine, BENCHMARK_DRUGS
from core.features import load_scaler_metadata, extract_features_bbb, extract_features_pgp
from gnn.sequential_transfer_gnn import (
    GINEEncoderTrunk,
    SequentialTransferGNN,
    load_pretrained_bbb_encoder,
)

_results: List[Tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    _results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def verify_artifacts():
    section("1. Directory & Artifact Integrity")
    data_dir = os.path.join(BASE_DIR, "data")
    models_dir = os.path.join(BASE_DIR, "models")
    results_dir = os.path.join(BASE_DIR, "results")

    check("data directory exists", os.path.isdir(data_dir))
    check("models directory exists", os.path.isdir(models_dir))
    check("results directory exists", os.path.isdir(results_dir))

    b3db_path = os.path.join(data_dir, "B3DB_classification.tsv")
    pgp_path = os.path.join(data_dir, "pgp_broccatelli.tab")
    check("B3DB dataset present", os.path.isfile(b3db_path), f"size: {os.path.getsize(b3db_path):,} bytes")
    check("Broccatelli dataset present", os.path.isfile(pgp_path), f"size: {os.path.getsize(pgp_path):,} bytes")

    bbb_m = os.path.join(models_dir, "bbb_xgboost.json")
    pgp_m = os.path.join(models_dir, "pgp_xgboost.json")
    check("Stage 1 BBB XGBoost model present", os.path.isfile(bbb_m))
    check("Stage 2 P-gp XGBoost model present", os.path.isfile(pgp_m))


def verify_scalers_and_engine():
    section("2. Scaler & Engine Initialization")
    models_dir = os.path.join(BASE_DIR, "models")
    bbb_s = os.path.join(models_dir, "bbb_scaler_metadata.json")
    pgp_s = os.path.join(models_dir, "pgp_scaler_metadata.json")

    m_bbb, s_bbb = load_scaler_metadata(bbb_s)
    check("BBB scaler has 6 features and positive stds", len(m_bbb) == 6 and all(s > 0 for s in s_bbb))

    m_pgp, s_pgp = load_scaler_metadata(pgp_s)
    check("P-gp scaler has 12 features and positive stds", len(m_pgp) == 12 and all(s > 0 for s in s_pgp))

    engine = SequentialCascadedEngine(models_dir=models_dir)
    check("SequentialCascadedEngine initializes successfully", engine is not None)
    return engine


def verify_sieve_logic(engine: SequentialCascadedEngine):
    section("3. Sequential Sieve Invariants & Behavioral Contracts")

    # A. Levetiracetam: Optimal CNS (Must pass Stage 1 and Stage 2)
    smi_keppra = BENCHMARK_DRUGS["Levetiracetam (Keppra)"]["smiles"]
    res_k = engine.evaluate_molecule(smi_keppra)
    check("Levetiracetam passes Stage 1 BBB sieve", res_k.stage1_pass, f"P(BBB+) = {res_k.stage1_prob_bbb:.3f}")
    check("Levetiracetam evaluates Stage 2 P-gp gate", res_k.stage2_evaluated)
    check("Levetiracetam evades P-gp efflux", res_k.stage2_pass, f"P(P-gp+) = {res_k.stage2_prob_pgp:.3f}")
    check("Levetiracetam disposition is CNS_PENETRANT", res_k.final_disposition == "CNS_PENETRANT")
    check("Levetiracetam composite CNS score > 0", res_k.composite_cns_score > 0.5)

    # B. Loperamide: The Paradoxical Drug (Must pass Stage 1, but get caught at Stage 2)
    smi_lop = BENCHMARK_DRUGS["Loperamide (Imodium)"]["smiles"]
    res_l = engine.evaluate_molecule(smi_lop)
    check("Loperamide passes Stage 1 BBB sieve", res_l.stage1_pass, f"P(BBB+) = {res_l.stage1_prob_bbb:.3f}")
    check("Loperamide evaluates Stage 2 P-gp gate", res_l.stage2_evaluated)
    check("Loperamide fails Stage 2 (Caught as substrate)", not res_l.stage2_pass, f"P(P-gp+) = {res_l.stage2_prob_pgp:.3f}")
    check("Loperamide disposition is PARADOXICAL_EFFLUX", res_l.final_disposition == "PARADOXICAL_EFFLUX")
    check("Loperamide composite score is zeroed out", res_l.composite_cns_score == 0.0)

    # C. Penicillin G: Peripheral Impermeant (Must fail Stage 1, Stage 2 MUST BE BYPASSED)
    smi_pen = BENCHMARK_DRUGS["Penicillin G"]["smiles"]
    res_p = engine.evaluate_molecule(smi_pen)
    check("Penicillin G fails Stage 1 BBB sieve", not res_p.stage1_pass, f"P(BBB+) = {res_p.stage1_prob_bbb:.3f}")
    check("Penicillin G BYPASSES Stage 2 P-gp evaluation", not res_p.stage2_evaluated)
    check("Penicillin G Stage 2 prob is None", res_p.stage2_prob_pgp is None)
    check("Penicillin G compute_saved is TRUE", res_p.compute_saved)
    check("Penicillin G disposition is PERIPHERAL_LOW_PERMEATION", res_p.final_disposition == "PERIPHERAL_LOW_PERMEATION")

    # D. Caffeine & Diazepam: Confirmed CNS
    smi_caf = BENCHMARK_DRUGS["Caffeine"]["smiles"]
    res_c = engine.evaluate_molecule(smi_caf)
    check("Caffeine is CNS_PENETRANT", res_c.final_disposition == "CNS_PENETRANT")

    smi_dzp = BENCHMARK_DRUGS["Diazepam (Valium)"]["smiles"]
    res_d = engine.evaluate_molecule(smi_dzp)
    check("Diazepam is CNS_PENETRANT", res_d.final_disposition == "CNS_PENETRANT")


def verify_sequential_transfer_gnn():
    section("4. Sequential Transfer Learning GNN Invariants")
    models_dir = os.path.join(BASE_DIR, "models")
    bbb_ckpt = os.path.join(models_dir, "gnn_single_bbb_random_encoded.pt")

    device = torch.device("cpu")
    check("Pre-trained BBB checkpoint exists", os.path.isfile(bbb_ckpt))

    trunk = load_pretrained_bbb_encoder(bbb_ckpt, device=device)
    check("Transferred encoder trunk loaded successfully", trunk is not None)
    check("Trunk has 3 GINEConv layers", len(trunk.convs) == 3)
    check("Trunk output channels == 256", trunk.hidden_channels == 256)

    # Instantiate SequentialTransferGNN
    model = SequentialTransferGNN(trunk=trunk, readout_dim=32, num_classes=2, freeze_trunk=True)
    check("SequentialTransferGNN initialized with frozen trunk", model is not None)

    # Check parameter freezing contract
    trunk_grad_status = [p.requires_grad for p in model.trunk.parameters()]
    head_grad_status = [p.requires_grad for p in model.head.parameters()]
    check("All trunk parameters are frozen (requires_grad=False)", not any(trunk_grad_status))
    check("Head parameters are trainable (requires_grad=True)", all(head_grad_status))

    # Test unfreezing
    model.unfreeze_trunk()
    check("Unfreezing successfully enables trunk gradients", all(p.requires_grad for p in model.trunk.parameters()))


def main():
    print("=" * 80)
    print("NOVELTY 2 SEQUENTIAL PIPELINE VERIFICATION SUITE")
    print("=" * 80)

    verify_artifacts()
    engine = verify_scalers_and_engine()
    verify_sieve_logic(engine)
    verify_sequential_transfer_gnn()

    print("\n" + "=" * 80)
    n_pass = sum(1 for _, p, _ in _results if p)
    n_total = len(_results)
    print(f"RESULT: {n_pass}/{n_total} checks passed")
    print("=" * 80)

    if n_pass != n_total:
        sys.exit(1)
    else:
        print("[SUCCESS] All Novelty 2 invariants verified successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
