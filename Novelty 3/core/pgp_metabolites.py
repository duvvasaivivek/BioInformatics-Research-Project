"""Audit independent metabolite evidence and evaluate locked P-gp cutoffs."""

from collections import defaultdict
import json

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import binomtest

from .audit import family_components
from .dataset import leakage_groups, load_dataset
from .learning import MODELS, calibrated_ensemble, device_for, metrics, score, sensitivity_threshold
from .pgp_baseline import fingerprint_predictions, load_baseline
from .sources import digest
from .storage import ROOT, write_json
from .validation import PILOT_INPUT, augmented_structures, measured_pairs

INPUT = ROOT / "data/input/pgp_metabolite_benchmark.json"
LOCK = MODELS / "pgp/metabolite_cutoffs.json"
REPORT = ROOT / "results/pgp_metabolite_evaluation.json"
TARGET = 0.90
NEIGHBOUR = 0.80
PROTOCOL = 1


def read_cases(data):
    cases = []
    seen = set()
    study_urls = set()
    for study in data["studies"].values():
        url = study["source"].rstrip("/").casefold()
        if not url.startswith("https://") or url in study_urls:
            raise ValueError("Each study requires one distinct canonical HTTPS citation.")
        study_urls.add(url)
        if study["source_overlap"] not in {"clear", "unresolved", "overlap"}:
            raise ValueError("Invalid source-overlap review status.")
        if not study["source_review"].strip():
            raise ValueError("Document the source-overlap review for each study.")
    for case in data["cases"]:
        parent_record = data["molecules"][case["parent"]]
        child_record = data["molecules"][case["metabolite"]]
        parent, child = parent_record["inchikey"], child_record["inchikey"]
        study = data["studies"][case["study"]]
        if parent == child or child in seen:
            raise ValueError("Each distinct metabolite needs one adjudicated record.")
        seen.add(child)
        label = case["label"]
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError("Use 1 for substrate, 0 for nonsubstrate or null for unknown/conflicting evidence.")
        if case["split"] not in {"development", "test", "exploratory"}:
            raise ValueError("Invalid metabolite cohort.")
        for field in ("assay", "evidence", "metabolism_source"):
            if not case[field].strip():
                raise ValueError(f"Case {case['metabolite']} is missing {field}.")
        reasons = []
        if any(not record["structure_source"].startswith("https://") for record in (parent_record, child_record)):
            raise ValueError("Provide traceable structure references for both compounds.")
        if parent_record.get("identity_note") or child_record.get("identity_note"):
            reasons.append("identity_review_pending")
        if label is None:
            reasons.append("unknown_or_conflicting_label")
        if study["source_overlap"] != "clear":
            reasons.append("publication_" + study["source_overlap"])
        if case["split"] == "exploratory":
            reasons.append("exploratory")
        cases.append({**case, "parent_inchikey": parent, "metabolite_inchikey": child,
                      "exclusions": reasons})
    return cases


def group_cases(cache, cases, prior_pairs, records, parsed):
    links = [{"parent_inchikey": a, "metabolite_inchikey": b} for a, b in cache["relationships"]]
    links.extend([*prior_pairs, *cases])
    known = {key for splits in cache["splits"].values() for keys in splits.values() for key in keys}
    previously_reviewed = {row[field] for row in prior_pairs for field in ("parent_inchikey", "metabolite_inchikey")}
    annotated = set().union(*(labels.keys() for labels in cache["labels"].values()))
    previously_reviewed.update(child for _, child in cache["relationships"] if child in annotated)
    candidates = {row[field] for row in cases for field in ("parent_inchikey", "metabolite_inchikey")}
    by_study = defaultdict(list)
    for row in cases:
        by_study[row["study"]].append(row["parent_inchikey"])
    for study_keys in by_study.values():
        links.extend({"parent_inchikey": study_keys[0], "metabolite_inchikey": key} for key in study_keys[1:])

    # Screen curated compounds against development, previously tested compounds and one another.
    neighbours = sorted(known | previously_reviewed | candidates)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {key: generator.GetFingerprint(parsed[key]) for key in neighbours}
    compare = [fps[key] for key in neighbours]
    for key in sorted(candidates):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[key], compare)
        links.extend({"parent_inchikey": key, "metabolite_inchikey": other}
                     for other, similarity in zip(neighbours, similarities)
                     if other != key and similarity >= NEIGHBOUR)
    groups = leakage_groups(records, family_components(links), parsed)

    seen_groups = {}
    for endpoint, splits in cache["splits"].items():
        for split, keys in splits.items():
            for key in keys:
                seen_groups.setdefault(groups[key], set()).add(f"{endpoint}:{split}")
    for key in previously_reviewed:
        seen_groups.setdefault(groups[key], set()).add("prior_metabolite_benchmark")
    active_splits = defaultdict(set)
    for row in cases:
        group = groups[row["parent_inchikey"]]
        row["group_id"] = group
        row["overlap"] = sorted(seen_groups.get(group, ()))
        if row["overlap"]:
            row["exclusions"].append("group_previously_used")
        if row["split"] != "exploratory":
            active_splits[group].add(row["split"])
    for row in cases:
        if len(active_splits[row["group_id"]]) > 1:
            row["exclusions"].append("group_crosses_new_splits")


def cohort():
    raw = INPUT.read_bytes()
    data = json.loads(raw)
    cache = load_dataset()
    registered = json.loads((ROOT / "data/processed/molecules.json").read_text(encoding="utf-8"))["molecules"]
    pilot_raw = PILOT_INPUT.read_bytes()
    pilot = json.loads(pilot_raw)
    cases = read_cases(data)
    prior_pairs = measured_pairs(pilot["studies"], pilot["molecules"])
    compounds = {**pilot["molecules"], **data["molecules"]}
    records, parsed = augmented_structures(cache, compounds, registered)
    group_cases(cache, cases, prior_pairs, records, parsed)
    return cache, records, cases, {"cohort_sha256": digest(raw), "prior_pilot_sha256": digest(pilot_raw)}


def eligible(cases, split):
    return sorted((case for case in cases if case["split"] == split and not case["exclusions"]),
                  key=lambda row: row["metabolite_inchikey"])


def audit_cohort():
    _, _, cases, _ = cohort()
    print(f"Reviewed {len(cases)} source records, including "
          f"{sum(case['label'] is None for case in cases)} unknown or conflicting labels.")
    for split in ("development", "test"):
        rows = eligible(cases, split)
        print(f"{split}: {len(rows)} eligible metabolites in {len({row['group_id'] for row in rows})} groups before BBB.")
    for case in cases:
        print(f"  {case['metabolite']}: " + (", ".join(case["exclusions"] + case["overlap"]) or "eligible"))


def frozen_models(cache, receipt):
    device = device_for("cpu")
    gnn = calibrated_ensemble(cache, "pgp", device)
    bbb = calibrated_ensemble(cache, "bbb", device)
    baseline, metadata, sha = load_baseline(cache, gnn[1]["target_sensitivity"])
    receipt.update({"protocol": PROTOCOL, "data_id": cache["data_id"], "target": TARGET,
                    "bbb_calibration": bbb[1], "pgp_gnn_calibration": gnn[1], "xgboost_sha256": sha})
    return device, bbb, gnn, baseline, metadata["calibration"]


def score_split(cache, records, rows, models):
    device, bbb, gnn, baseline, baseline_cal = models
    keys = [row["metabolite_inchikey"] for row in rows]
    scores = score(cache, "bbb", keys, bbb, device)
    passed = [row for row in rows if scores[row["metabolite_inchikey"]][0] >= bbb[1]["threshold"]]
    keys = [row["metabolite_inchikey"] for row in passed]
    gnn_scores = score(cache, "pgp", keys, gnn, device)
    probabilities = {
        "gnn": np.asarray([gnn_scores[key][0] for key in keys]),
        "xgboost": fingerprint_predictions(baseline, baseline_cal, [records[key]["canonical_smiles"] for key in keys]),
    }
    return passed, np.asarray([row["label"] for row in passed], dtype=int), probabilities


def fit_cutoffs():
    cache, records, cases, receipt = cohort()
    dev = eligible(cases, "development")
    if {row["label"] for row in dev} != {0, 1}:
        raise ValueError("Need both verified classes in development. Run 'pgp-metabolites audit'.")
    models = frozen_models(cache, receipt)
    passed, y, probs = score_split(cache, records, dev, models)
    if set(y) != {0, 1}:
        raise ValueError("Development needs both measured classes after the frozen BBB gate.")
    cutoffs = {name: sensitivity_threshold(y, p, TARGET) for name, p in probs.items()}
    result = {"receipt": receipt, "cutoffs": cutoffs,
              "development_counts": {name: metrics(y, p, cutoffs[name]) for name, p in probs.items()},
              "bbb_rejected": len(dev) - len(passed)}
    write_json(LOCK, result)
    print(f"Locked both P-gp cutoffs at a {TARGET:.0%} development sensitivity target: {LOCK}")


def paired_differences(y, group_ids, challenger, reference):
    """Paired cluster bootstrap; positive deltas mean better detection or fewer false flags."""
    groups, inverse = np.unique(group_ids, return_inverse=True)
    if any(len(np.unique(inverse[y == label])) < 2 for label in (0, 1)):
        return None
    totals = np.zeros((len(groups), 4), dtype=int)
    difference = challenger.astype(int) - reference.astype(int)
    np.add.at(totals, inverse, np.column_stack(((y == 1), (y == 0),
                                                difference * (y == 1), -difference * (y == 0))).astype(int))
    rng = np.random.default_rng(2026)
    samples = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=10000) @ totals
    samples = samples[(samples[:, 0] > 0) & (samples[:, 1] > 0)]
    interval = np.quantile(samples[:, 2:] / samples[:, :2], [0.025, 0.975], axis=0)
    improved = int(np.sum(totals[:, 3] > 0))
    worsened = int(np.sum(totals[:, 3] < 0))
    p_value = binomtest(improved, improved + worsened, alternative="greater").pvalue if improved + worsened else 1.0
    return {"substrate_detection_rate_delta_95ci": interval[:, 0].tolist(),
            "false_flag_rate_reduction_95ci": interval[:, 1].tolist(),
            "false_flag_group_sign_test_p": p_value}


def evaluate_cutoffs():
    if not LOCK.exists():
        raise ValueError("Run 'pgp-metabolites calibrate' to lock development cutoffs first.")
    saved = json.loads(LOCK.read_text(encoding="utf-8"))
    cache, records, cases, receipt = cohort()
    test = eligible(cases, "test")
    if {row["label"] for row in test} != {0, 1}:
        raise ValueError("Need both verified classes in the locked test cohort. Run 'pgp-metabolites audit'.")
    models = frozen_models(cache, receipt)
    if receipt != saved["receipt"]:
        raise ValueError("Benchmark, checkpoints or calibration changed after cutoff selection.")
    passed, y, probs = score_split(cache, records, test, models)
    deployed = models[2][1]["threshold"]
    cutoffs = {"current_gnn": deployed, "metabolite_gnn": saved["cutoffs"]["gnn"],
               "metabolite_xgboost": saved["cutoffs"]["xgboost"]}
    probabilities = {"current_gnn": probs["gnn"], "metabolite_gnn": probs["gnn"],
                     "metabolite_xgboost": probs["xgboost"]}
    results = {name: metrics(y, p, cutoffs[name]) for name, p in probabilities.items()}
    reference = probabilities["current_gnn"] >= deployed
    uncertainty = {
        name: paired_differences(y, [row["group_id"] for row in passed], probabilities[name] >= cutoffs[name], reference)
        for name in ("metabolite_gnn", "metabolite_xgboost")
    }
    challenger, current = results["metabolite_xgboost"], results["current_gnn"]
    ci = uncertainty["metabolite_xgboost"]
    better = (ci is not None and challenger["tp"] >= current["tp"] and challenger["fp"] < current["fp"]
              and ci["substrate_detection_rate_delta_95ci"][0] >= 0
              and ci["false_flag_rate_reduction_95ci"][0] > 0
              and ci["false_flag_group_sign_test_p"] < 0.05)
    report = {
        "lock": saved, "cutoffs": cutoffs, "metrics": results, "uncertainty": uncertainty,
        "eligible_test_molecules": len(test), "bbb_rejected": len(test) - len(passed),
        "test_unknown_or_conflicting": sum(row["label"] is None and row["split"] == "test" for row in cases),
        "test_groups_after_bbb": len({row["group_id"] for row in passed}),
        "decision": "xgboost_candidate" if better else "keep_current_gnn",
        "decision_evidence": "Paired 95% group bootstrap intervals against the deployed GNN. "
                             "The one-sided sign test across groups guards against a bootstrap interval "
                             "that looks conclusive when only a few groups change. "
                             "A candidate must retain detection and reduce false flags; "
                             "an unsupported gain keeps the deployed GNN.",
        "cases": [row for row in cases if row["split"] == "test"],
    }
    write_json(REPORT, report)
    print(f"Tested {len(passed)} metabolites after BBB. Decision: {report['decision']}. Report: {REPORT}")
