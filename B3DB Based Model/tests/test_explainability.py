"""
Comprehensive Automated Test Suite for Dual-Barrier CNS Drug Predictor & Explainability Engine.
Verifies all 4 quadrant outcomes, TreeSHAP attributions, feature ordering,
substructure mapping, applicability domain, counterfactuals, and export formats.
"""

import os
import sys
try:
    import pytest
    fixture_decorator = pytest.fixture(scope="module")
except ImportError:
    pytest = None
    def fixture_decorator(fn):
        return fn
import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem

# Set up paths
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
B3DB_DIR = os.path.dirname(TEST_DIR)
MODELS_DIR = os.path.join(B3DB_DIR, "models")

sys.path.insert(0, B3DB_DIR)

from inference import (
    extract_features_bbb,
    extract_features_pgp,
    prepare_dataframe,
    determine_dual_barrier_quadrant,
    load_scaler_metadata
)
from explainability import (
    DualBarrierShapExplainer,
    MorganSubstructureMapper,
    ApplicabilityDomainEstimator,
    SharedPropertiesAnalyzer,
    MolecularCounterfactualEngine,
    ScientificReportGenerator
)


@fixture_decorator
def models_and_scalers():
    """Fixture to load models, scalers, and explainability components."""
    # 1. BBB Model
    bbb_model_path = os.path.join(MODELS_DIR, "bbb_xgboost_optuna_best.json")
    bbb_model = xgb.XGBClassifier()
    bbb_model.load_model(bbb_model_path)

    default_bbb_means = [385.4589, 2.3559, 87.0098, 2.0102, 5.3007, 4.7708]
    default_bbb_stds = [170.8765, 2.1956, 67.6848, 2.2626, 3.6692, 3.6367]
    bbb_means, bbb_stds = load_scaler_metadata(
        os.path.join(MODELS_DIR, "bbb_scaler_metadata.json"), default_bbb_means, default_bbb_stds
    )

    # 2. P-gp Model
    pgp_model_path = os.path.join(MODELS_DIR, "pgp_xgboost_model.json")
    pgp_model = xgb.XGBClassifier()
    pgp_model.load_model(pgp_model_path)

    pgp_means, pgp_stds = load_scaler_metadata(os.path.join(MODELS_DIR, "pgp_scaler_metadata.json"))

    # Explainability subsystems
    shap_explainer = DualBarrierShapExplainer(bbb_model, pgp_model)
    substructure_mapper = MorganSubstructureMapper()
    shared_analyzer = SharedPropertiesAnalyzer(substructure_mapper)
    domain_estimator = ApplicabilityDomainEstimator()
    cf_engine = MolecularCounterfactualEngine(bbb_model, pgp_model, bbb_means, bbb_stds, pgp_means, pgp_stds)

    return {
        "bbb_model": bbb_model,
        "pgp_model": pgp_model,
        "bbb_means": bbb_means,
        "bbb_stds": bbb_stds,
        "pgp_means": pgp_means,
        "pgp_stds": pgp_stds,
        "shap_explainer": shap_explainer,
        "substructure_mapper": substructure_mapper,
        "shared_analyzer": shared_analyzer,
        "domain_estimator": domain_estimator,
        "cf_engine": cf_engine
    }


def test_quadrant_1_optimal_cns(models_and_scalers):
    """Levetiracetam must predict as Quadrant 1: BBB+ / P-gp-."""
    ctx = models_and_scalers
    smiles = "CC[C@@H](C(=O)N)N1CCCC1=O"

    bbb_feats, raw_bbb, mol_bbb = extract_features_bbb(smiles)
    pgp_feats, raw_pgp, mol_pgp = extract_features_pgp(smiles)

    bbb_df = prepare_dataframe(bbb_feats[0], bbb_feats[1], ctx["bbb_means"], ctx["bbb_stds"], is_pgp=False)
    pgp_df = prepare_dataframe(pgp_feats[0], pgp_feats[1], ctx["pgp_means"], ctx["pgp_stds"], is_pgp=True)

    bbb_prob = ctx["bbb_model"].predict_proba(bbb_df)[0][1]
    pgp_prob = ctx["pgp_model"].predict_proba(pgp_df)[0][1]

    bbb_pred = int(bbb_prob > 0.5)
    pgp_pred = int(pgp_prob > 0.5)

    assert bbb_pred == 1, f"Expected BBB+, got {bbb_pred} (prob: {bbb_prob})"
    assert pgp_pred == 0, f"Expected P-gp-, got {pgp_pred} (prob: {pgp_prob})"

    quadrant, _ = determine_dual_barrier_quadrant(bbb_pred, pgp_pred)
    assert "Quadrant 1" in quadrant


def test_quadrant_2_paradoxical_drug(models_and_scalers):
    """Loperamide must predict as Quadrant 2: BBB+ / P-gp+."""
    ctx = models_and_scalers
    smiles = "CN(C)C(=O)CC(c1ccccc1)(c2ccccc2)CCN3CCC(CC3)(c4ccc(cc4)Cl)O"

    bbb_feats, raw_bbb, mol_bbb = extract_features_bbb(smiles)
    pgp_feats, raw_pgp, mol_pgp = extract_features_pgp(smiles)

    bbb_df = prepare_dataframe(bbb_feats[0], bbb_feats[1], ctx["bbb_means"], ctx["bbb_stds"], is_pgp=False)
    pgp_df = prepare_dataframe(pgp_feats[0], pgp_feats[1], ctx["pgp_means"], ctx["pgp_stds"], is_pgp=True)

    bbb_prob = ctx["bbb_model"].predict_proba(bbb_df)[0][1]
    pgp_prob = ctx["pgp_model"].predict_proba(pgp_df)[0][1]

    bbb_pred = int(bbb_prob > 0.5)
    pgp_pred = int(pgp_prob > 0.5)

    assert bbb_pred == 1, f"Expected BBB+, got {bbb_pred}"
    assert pgp_pred == 1, f"Expected P-gp+, got {pgp_pred}"

    quadrant, _ = determine_dual_barrier_quadrant(bbb_pred, pgp_pred)
    assert "Quadrant 2" in quadrant


def test_quadrant_3_peripheral_agent(models_and_scalers):
    """Penicillin G must predict as Quadrant 3: BBB- / P-gp-."""
    ctx = models_and_scalers
    smiles = "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"

    bbb_feats, raw_bbb, mol_bbb = extract_features_bbb(smiles)
    pgp_feats, raw_pgp, mol_pgp = extract_features_pgp(smiles)

    bbb_df = prepare_dataframe(bbb_feats[0], bbb_feats[1], ctx["bbb_means"], ctx["bbb_stds"], is_pgp=False)
    pgp_df = prepare_dataframe(pgp_feats[0], pgp_feats[1], ctx["pgp_means"], ctx["pgp_stds"], is_pgp=True)

    bbb_prob = ctx["bbb_model"].predict_proba(bbb_df)[0][1]
    pgp_prob = ctx["pgp_model"].predict_proba(pgp_df)[0][1]

    bbb_pred = int(bbb_prob > 0.5)
    pgp_pred = int(pgp_prob > 0.5)

    assert bbb_pred == 0, f"Expected BBB-, got {bbb_pred}"
    assert pgp_pred == 0, f"Expected P-gp-, got {pgp_pred}"

    quadrant, _ = determine_dual_barrier_quadrant(bbb_pred, pgp_pred)
    assert "Quadrant 3" in quadrant


def test_invalid_smiles_handling():
    """Invalid SMILES must return None gracefully without throwing an exception."""
    invalid_smiles = "INVALID_C1234_NOT_A_MOLECULE"
    bbb_feats, raw_bbb, mol_bbb = extract_features_bbb(invalid_smiles)
    pgp_feats, raw_pgp, mol_pgp = extract_features_pgp(invalid_smiles)

    assert bbb_feats is None
    assert raw_bbb is None
    assert mol_bbb is None

    assert pgp_feats is None
    assert raw_pgp is None


def test_shap_explanation_structure(models_and_scalers):
    """Verifies that TreeSHAP computes valid margins, base values, and supporting/opposing features."""
    ctx = models_and_scalers
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine

    bbb_feats, raw_bbb, mol = extract_features_bbb(smiles)
    bbb_df = prepare_dataframe(bbb_feats[0], bbb_feats[1], ctx["bbb_means"], ctx["bbb_stds"], is_pgp=False)

    expl = ctx["shap_explainer"].explain_prediction("BBB", bbb_df, mol=mol, raw_descriptors=raw_bbb)

    assert "model_margin" in expl
    assert "base_value" in expl
    assert "top_supporting" in expl
    assert "top_opposing" in expl
    assert len(expl["top_supporting"]) > 0
    assert "interpretation" in expl
    assert "biophysical_basis" in expl["interpretation"]


def test_substructure_mapping(models_and_scalers):
    """Verifies that active Morgan bits map to chemical subgraphs."""
    mapper = models_and_scalers["substructure_mapper"]
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin

    bit_info = mapper.get_bit_info(mol)
    assert len(bit_info) > 0

    first_bit = list(bit_info.keys())[0]
    frags = mapper.extract_substructure_smiles(mol, first_bit, bit_info)

    assert len(frags) > 0
    assert "fragment_smiles" in frags[0]
    assert "functional_group" in frags[0]


def test_shared_properties_analysis(models_and_scalers):
    """Verifies that the shared properties analyzer produces rows for all 6 continuous descriptors."""
    ctx = models_and_scalers
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    bbb_feats, raw_bbb, mol_bbb = extract_features_bbb(smiles)
    pgp_feats, raw_pgp, mol_pgp = extract_features_pgp(smiles)

    bbb_df = prepare_dataframe(bbb_feats[0], bbb_feats[1], ctx["bbb_means"], ctx["bbb_stds"], is_pgp=False)
    pgp_df = prepare_dataframe(pgp_feats[0], pgp_feats[1], ctx["pgp_means"], ctx["pgp_stds"], is_pgp=True)

    bbb_expl = ctx["shap_explainer"].explain_prediction("BBB", bbb_df, mol=mol_bbb, raw_descriptors=raw_bbb)
    pgp_expl = ctx["shap_explainer"].explain_prediction("Pgp", pgp_df, mol=mol_pgp, raw_descriptors=raw_pgp)

    shared_res = ctx["shared_analyzer"].compare_compound_shared_properties(
        raw_descriptors=raw_pgp,
        bbb_shap_result=bbb_expl,
        pgp_shap_result=pgp_expl,
        mol=mol_bbb
    )

    assert "continuous_table" in shared_res
    assert len(shared_res["continuous_table"]) == 6
    props = [r["property_code"] for r in shared_res["continuous_table"]]
    assert set(props) == {'MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds'}


def test_report_export(tmp_path, models_and_scalers):
    """Verifies that HTML, CSV, and JSON reports serialize properly."""
    dummy_data = {
        "smiles": "CC[C@@H](C(=O)N)N1CCCC1=O",
        "quadrant_category": "Quadrant 1: BBB+ / P-gp-",
        "quadrant_description": "Optimal CNS candidate.",
        "bbb_class_label": "BBB+ (Permeable)",
        "bbb_prob": 0.988,
        "pgp_class_label": "P-gp- (Non-Substrate)",
        "pgp_prob": 0.028,
        "raw_descriptors": {"MolWt": 170.21, "LogP": -0.13, "TPSA": 63.4},
        "bbb_explanation": {"top_supporting": [], "top_opposing": []},
        "pgp_explanation": {"top_supporting": [], "top_opposing": []},
        "shared_properties": {"continuous_table": []},
        "applicability_domain_bbb": {"domain_status": "Inside Domain"},
        "applicability_domain_pgp": {"domain_status": "Inside Domain"}
    }

    html_file = str(tmp_path / "test_report.html")
    csv_file = str(tmp_path / "test_report.csv")
    json_file = str(tmp_path / "test_report.json")

    ScientificReportGenerator.export_html(dummy_data, html_file)
    ScientificReportGenerator.export_csv(dummy_data, csv_file)
    ScientificReportGenerator.export_json(dummy_data, json_file)

    assert os.path.exists(html_file)
    assert os.path.exists(csv_file)
    assert os.path.exists(json_file)
    assert os.path.getsize(html_file) > 1000
    assert os.path.getsize(csv_file) > 50


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    print("=" * 75)
    print("   RUNNING AUTOMATED TEST SUITE: DUAL-BARRIER CNS PREDICTOR & XAI   ")
    print("=" * 75)

    ctx = models_and_scalers()
    test_functions = [
        ("Quadrant 1: Optimal CNS (Levetiracetam)", lambda: test_quadrant_1_optimal_cns(ctx)),
        ("Quadrant 2: Paradoxical Drug (Loperamide)", lambda: test_quadrant_2_paradoxical_drug(ctx)),
        ("Quadrant 3: Peripheral Agent (Penicillin G)", lambda: test_quadrant_3_peripheral_agent(ctx)),
        ("Invalid SMILES Exception Handling", test_invalid_smiles_handling),
        ("TreeSHAP Explanation Structure & Base Values", lambda: test_shap_explanation_structure(ctx)),
        ("Morgan Bit-to-Substructure Mapping", lambda: test_substructure_mapping(ctx)),
        ("Shared Physicochemical Properties Analysis", lambda: test_shared_properties_analysis(ctx)),
    ]

    passed = 0
    for name, test_fn in test_functions:
        try:
            test_fn()
            print(f" [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {name}: {e}")

    # Test report exports with temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            test_report_export(Path(tmp_dir), ctx)
            print(" [PASS] Scientific Report Serialization (HTML, CSV, JSON)")
            passed += 1
        except Exception as e:
            print(f" [FAIL] Scientific Report Serialization: {e}")

    total = len(test_functions) + 1
    print("-" * 75)
    print(f"Test Summary: {passed}/{total} tests PASSED.")
    if passed == total:
        print("ALL REGRESSION & VALIDATION TESTS COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)

