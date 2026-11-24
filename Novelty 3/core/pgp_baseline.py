"""Fit an independent MACCS XGBoost baseline and compare frozen P-gp models."""

import json

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from scipy.special import expit
from sklearn.metrics import average_precision_score
import xgboost as xgb

from .dataset import load_dataset
from .learning import MODELS, calibrated_ensemble, device_for, fit_score_calibration, metrics, score
from .sources import digest
from .storage import ROOT, atomic_bytes, write_json

BASELINE_PATH = MODELS / "pgp" / "maccs_xgboost.ubj"
COMPARISON_PATH = ROOT / "results" / "pgp_comparison.json"
VERSION = 1
MACCS_BITS = 167
THREADS = 4
ROUNDS = 500
PATIENCE = 30


def fingerprint(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Cannot fingerprint an invalid molecular structure.")
    row = np.empty(MACCS_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(molecule), row)
    return row


def fingerprint_predictions(model, calibration, structures):
    if not structures:
        return np.empty(0)
    matrix = xgb.DMatrix(np.stack([fingerprint(smiles) for smiles in structures]), nthread=THREADS)
    margins = model.predict(matrix, output_margin=True).astype(np.float64)
    if not np.isfinite(margins).all():
        raise ValueError("The P-gp fingerprint model produced non-finite scores.")
    return expit(calibration["slope"] * margins + calibration["intercept"])


def matrices(cache, keys_by_split):
    source = ROOT / "data" / "processed" / "molecules.json"
    molecules = json.loads(source.read_text(encoding="utf-8"))["molecules"]
    labels = cache["labels"]["pgp"]
    fingerprints = {key: fingerprint(molecules[key]["canonical_smiles"])
                    for key in set().union(*keys_by_split.values())}
    return {
        name: xgb.DMatrix(np.stack([fingerprints[key] for key in keys]),
                          label=[labels[key] for key in keys], nthread=THREADS)
        for name, keys in keys_by_split.items() if keys
    }


def validation_ap(prediction, dataset):
    return "average_precision", float(average_precision_score(dataset.get_label(), prediction))


def fit_baseline():
    cache = load_dataset()
    gnn_path = MODELS / "pgp" / "calibration.json"
    if not gnn_path.exists():
        raise ValueError("Run 'calibrate pgp' to establish the GNN sensitivity target first.")
    gnn_calibration = json.loads(gnn_path.read_text(encoding="utf-8"))
    if gnn_calibration["data_id"] != cache["data_id"]:
        raise ValueError("The P-gp GNN calibration belongs to a different dataset.")
    target = gnn_calibration["target_sensitivity"]

    parts = cache["splits"]["pgp"]
    data = matrices(cache, {name: parts[name] for name in ("train", "validation", "calibration")})
    train_y = data["train"].get_label()
    positive = float(train_y.sum())
    params = {
        "objective": "binary:logistic", "tree_method": "hist", "nthread": THREADS, "seed": 2026,
        "max_depth": 4, "eta": 0.05, "subsample": 0.85, "colsample_bytree": 0.85,
        "scale_pos_weight": (len(train_y) - positive) / positive, "disable_default_eval_metric": 1,
    }
    print(f"Fitting MACCS XGBoost on {len(train_y)} P-gp training compounds (CPU).", flush=True)
    stop = xgb.callback.EarlyStopping(rounds=PATIENCE, maximize=True, save_best=True)
    model = xgb.train(params, data["train"], num_boost_round=ROUNDS,
                      evals=[(data["validation"], "validation")],
                      custom_metric=validation_ap, callbacks=[stop], verbose_eval=25)
    logits = model.predict(data["calibration"], output_margin=True)
    calibration = fit_score_calibration(logits.astype(np.float64), data["calibration"].get_label(), target)
    metadata = {"version": VERSION, "data_id": cache["data_id"], "params": params,
                "num_boost_round": ROUNDS, "patience": PATIENCE,
                "best_round": int(model.best_iteration) + 1, "validation_ap": float(model.best_score),
                "calibration": calibration}
    model.set_attr(novelty3=json.dumps(metadata, allow_nan=False))
    atomic_bytes(BASELINE_PATH, bytes(model.save_raw(raw_format="ubj")))
    print(f"Best validation AP {metadata['validation_ap']:.4f} at round {metadata['best_round']}.")
    print(BASELINE_PATH)


def load_baseline(cache, target):
    if not BASELINE_PATH.exists():
        raise ValueError("Run 'fit-pgp-baseline' first.")
    raw = BASELINE_PATH.read_bytes()
    model = xgb.Booster(params={"nthread": THREADS})
    model.load_model(bytearray(raw))
    attr = model.attr("novelty3")
    if attr is None:
        raise ValueError("P-gp fingerprint model is missing its training metadata.")
    metadata = json.loads(attr)
    if metadata["data_id"] != cache["data_id"] or metadata["version"] != VERSION:
        raise ValueError("P-gp fingerprint model is stale. Run 'fit-pgp-baseline' on the current data.")
    if metadata["calibration"]["target_sensitivity"] != target:
        raise ValueError("P-gp models have different calibration targets. Rerun 'fit-pgp-baseline'.")
    return model, metadata, digest(raw)


def comparison_cohorts(cache, bbb_scores, bbb_cutoff):
    parts = cache["splits"]["pgp"]
    metabolites = sorted({child for _, child in cache["relationships"]} & cache["labels"]["pgp"].keys())
    groups = cache["groups"]
    bbb_development_groups = {groups[key] for part in ("train", "validation", "calibration")
                              for key in cache["splits"]["bbb"][part]}

    def eligible_after_bbb(keys):
        return [key for key in keys if groups[key] not in bbb_development_groups
                and bbb_scores[key][0] >= bbb_cutoff]

    return {
        "validation": parts["validation"],
        "calibration": parts["calibration"],
        "endpoint_test": parts["test"],
        "endpoint_test_after_bbb": eligible_after_bbb(parts["test"]),
        "reserved_metabolite_annotations": metabolites,
        "reserved_metabolite_annotations_after_bbb": eligible_after_bbb(metabolites),
    }


def compare_pgp(choice):
    cache = load_dataset()
    device = device_for(choice)
    pgp = calibrated_ensemble(cache, "pgp", device)
    bbb = calibrated_ensemble(cache, "bbb", device)
    target = pgp[1]["target_sensitivity"]
    baseline, metadata, baseline_sha = load_baseline(cache, target)
    metabolites = {child for _, child in cache["relationships"]} & cache["labels"]["pgp"].keys()
    bbb_keys = sorted(metabolites | set(cache["splits"]["pgp"]["test"]))
    bbb_scores = score(cache, "bbb", bbb_keys, bbb, device)
    cohorts = comparison_cohorts(cache, bbb_scores, bbb[1]["threshold"])
    pgp_keys = sorted(set().union(*(keys for name, keys in cohorts.items() if name != "calibration")))
    gnn_predictions = score(cache, "pgp", pgp_keys, pgp, device)
    matrix = matrices(cache, {"predict": pgp_keys})["predict"]
    baseline_cal = metadata["calibration"]
    margins = baseline.predict(matrix, output_margin=True)
    if not np.isfinite(margins).all():
        raise ValueError("The P-gp fingerprint model produced non-finite scores.")
    baseline_predictions = dict(zip(
        pgp_keys, expit(baseline_cal["slope"] * margins.astype(np.float64) + baseline_cal["intercept"])))
    results = {}
    for name, keys in cohorts.items():
        if name == "calibration":
            # Use the predictions that selected the threshold. Rebatching GPU
            # inference can move the threshold-defining example by a few ULPs.
            results[name] = {
                "groups": len({cache["groups"][key] for key in keys}),
                "gnn": pgp[1]["calibration_metrics"],
                "maccs_xgboost": baseline_cal["calibration_metrics"],
            }
            continue
        labels = [cache["labels"]["pgp"][key] for key in keys]
        results[name] = {
            "groups": len({cache["groups"][key] for key in keys}),
            "gnn": metrics(labels, [gnn_predictions[key][0] for key in keys], pgp[1]["threshold"]),
            "maccs_xgboost": metrics(labels, [baseline_predictions[key] for key in keys],
                                    baseline_cal["threshold"]),
        }
    report = {
        "data_id": cache["data_id"], "baseline_sha256": baseline_sha, "baseline": metadata,
        "pgp_gnn_calibration": pgp[1], "bbb_gnn_calibration": bbb[1],
        "scope": "Existing splits are a retrospective comparison. Validation drives early stopping. "
                 "Both cutoffs target the same calibration sensitivity; realized sensitivity can differ. "
                 "After-BBB results exclude groups used for BBB development. "
                 "Confirm model selection on new source-verified families.",
        "cohorts": results,
    }
    write_json(COMPARISON_PATH, report)
    print(f"Both cutoffs target {target:.0%} substrate sensitivity on the same calibration split.")
    for cohort, row in results.items():
        print(f"[{cohort}] n={row['gnn']['n']}")
        for name in ("gnn", "maccs_xgboost"):
            result = row[name]
            print(f"  {name}: caught {result['tp']}/{result['positive']} substrates; "
                  f"flagged {result['fp']}/{result['fp'] + result['tn']} nonsubstrates; "
                  f"ROC AUC {result['roc_auc']}")
    print(COMPARISON_PATH)
