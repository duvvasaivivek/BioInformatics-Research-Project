"""Compare frozen predictions with traceable paired brain-distribution measurements."""

import json
import math
from statistics import median

from rdkit import Chem
from rdkit.Chem import inchi

from .audit import family_components
from .dataset import ENDPOINTS, leakage_groups, load_dataset, molecule_graph
from .learning import calibrated_ensemble, device_for
from .screening import rank_candidates, sequential_scores
from .sources import digest
from .storage import ROOT, write_json

PILOT_INPUT = ROOT / "data" / "input" / "paired_validation.json"


def positive_measurement(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Paired validation requires positive, uncensored numeric measurements.")
    return value


def measured_pairs(studies, compounds):
    pairs = {}
    for study in studies:
        parent = study["parent"]
        for obs in study["observations"]:
            if "ratio" in obs:
                values = {name: positive_measurement(value) for name, value in obs["ratio"].items()}
            else:
                if obs["brain"].keys() != obs["systemic"].keys():
                    raise ValueError("Brain and systemic observations must identify the same compounds.")
                values = {name: positive_measurement(value) / positive_measurement(obs["systemic"][name])
                          for name, value in obs["brain"].items()}
            for child in sorted(values.keys() - {parent}):
                key = (compounds[parent]["inchikey"], compounds[child]["inchikey"])
                row = pairs.setdefault(key, {
                    "parent": parent, "metabolite": child, "study_id": study["id"],
                    "parent_inchikey": key[0], "metabolite_inchikey": key[1], "measurements": [],
                })
                if parent == child or key[0] == key[1] or row["study_id"] != study["id"]:
                    raise ValueError("Each distinct pair must have one designated study for the pilot comparison.")
                row["measurements"].append({"condition": obs["condition"], "parent_ratio": values[parent],
                                            "metabolite_ratio": values[child],
                                            "fold_vs_parent": values[child] / values[parent]})
    for row in pairs.values():
        fold = median([measurement["fold_vs_parent"] for measurement in row["measurements"]])
        row["median_measured_fold_vs_parent"] = fold
        row["higher_measured_ratio"] = fold > 1
    return list(pairs.values())


def augmented_structures(cache, compounds, registered):
    records = {key: registered[key] for key in cache["groups"]}
    for row in compounds.values():
        key = row["inchikey"]
        smiles = row.get("smiles") or registered[key]["canonical_smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or inchi.MolToInchiKey(mol, options="/W1") != key:
            raise ValueError(f"Structure does not match curated InChIKey {key}.")
        if key not in records:
            records[key] = {"canonical_smiles": Chem.MolToSmiles(mol), "connectivity_key": key.split("-")[0]}
            cache["graphs"][key] = molecule_graph(mol)
    parsed = {key: Chem.MolFromSmiles(row["canonical_smiles"]) for key, row in records.items()}
    return records, parsed


def holdout_groups(cache, rows, compounds, registered):
    records, parsed = augmented_structures(cache, compounds, registered)
    # New documented pairs can bridge pre-existing groups. Evaluate the whole merged component.
    edges = [{"parent_inchikey": parent, "metabolite_inchikey": child}
             for parent, child in cache["relationships"]]
    edges.extend(rows)
    families = family_components(edges)
    groups = leakage_groups(records, families, parsed)
    fitted = {}
    for endpoint in ENDPOINTS:
        for split in ("train", "validation", "calibration"):
            for key in cache["splits"][endpoint][split]:
                fitted.setdefault(groups[key], set()).add(f"{endpoint}:{split}")
    for row in rows:
        row["group_id"] = groups[row["parent_inchikey"]]
        row["model_development_overlap"] = sorted(fitted.get(row["group_id"], ()))


def validate_pairs(choice, budgets):
    raw = PILOT_INPUT.read_bytes()
    pilot = json.loads(raw)
    cache = load_dataset()
    registered = json.loads((ROOT / "data" / "processed" / "molecules.json").read_text(encoding="utf-8"))
    studies = pilot["studies"]
    rows = measured_pairs(studies, pilot["molecules"])
    holdout_groups(cache, rows, pilot["molecules"], registered["molecules"])
    device = device_for(choice)
    ensembles = {task: calibrated_ensemble(cache, task, device) for task in ENDPOINTS}
    ids = sorted({row[field] for row in rows for field in ("parent_inchikey", "metabolite_inchikey")})
    scores = sequential_scores(cache, ids, ensembles, device)

    studies_by_id = {study["id"]: study for study in studies}
    for row in rows:
        parent, child = scores[row["parent_inchikey"]], scores[row["metabolite_inchikey"]]
        row.update({f"parent_{field}": value for field, value in parent.items()})
        row.update({f"metabolite_{field}": value for field, value in child.items()})
        row["parent_relative_support"] = (child["support"] - parent["support"]
                                          if child["stage"] == "pass" else None)
        evidence = studies_by_id[row["study_id"]].get("pgp_limitation", {})
        limited = set(evidence.get("fold_increase", ())) | set(evidence.get("reported_limited", ()))
        for role in ("parent", "metabolite"):
            stage = row[f"{role}_stage"]
            row[f"{role}_pgp_control"] = (
                None if row[role] not in limited else "not_reached" if stage == "bbb_rejected"
                else "flagged" if stage == "pgp_flagged" else "not_flagged")

    held_out = [row for row in rows if not row["model_development_overlap"]]
    picks = rank_candidates(held_out, group_field="group_id")
    comparison = {}
    for policy, ranked in picks.items():
        comparison[policy] = []
        for budget in budgets:
            selected = ranked[:budget]
            comparison[policy].append({
                "budget": budget, "selected": len(selected),
                "higher_measured_ratio": sum(row["higher_measured_ratio"] for row in selected),
                "pairs": [[row["parent"], row["metabolite"]] for row in selected],
            })
    report = {
        "status": "exploratory_frozen_model_pilot",
        "data_id": cache["data_id"], "pilot_sha256": digest(raw),
        "calibration": {task: ensemble[1] for task, ensemble in ensembles.items()},
        "measurement_rule": "Compare measured brain-to-systemic ratios within each study condition. "
                            "A higher ratio means the median within-condition metabolite/parent fold exceeds 1. "
                            "This is a descriptive comparison with no statistical superiority cutoff.",
        "holdout_rule": "Exclude family, connectivity or scaffold groups used for either model's training, "
                        "early stopping or calibration. Publication-level overlap with upstream data can remain.",
        "budget_rule": "Compare the same gated policies at the same maximum budget, with one pick per merged group. "
                       "Record shortfalls. The outcome is higher measured brain distribution, with P-gp controls "
                       "reported separately. Studies mix animal species and brain-distribution readouts.",
        "molecules": pilot["molecules"], "studies": studies, "pairs": rows,
        "eligible_pairs": len(held_out), "eligible_groups": len({row["group_id"] for row in held_out}),
        "eligible_higher_measured_ratio": sum(row["higher_measured_ratio"] for row in held_out),
        "fixed_budget": comparison,
    }
    output = ROOT / "results" / "paired_validation.json"
    write_json(output, report)
    print(f"Compared {len(rows)} measured pairs. {len(held_out)} in {report['eligible_groups']} groups "
          "qualify for the frozen-model comparison.")
    for row in rows:
        scope = "overlap" if row["model_development_overlap"] else "held out"
        print(f"  {row['parent']} -> {row['metabolite']}: {row['median_measured_fold_vs_parent']:.2f}x "
              f"measured ratio; {row['parent_stage']} -> {row['metabolite_stage']} ({scope})")
    for policy, result in comparison.items():
        print(f"  {policy}: " + ", ".join(f"budget {item['budget']}: {item['higher_measured_ratio']} "
                                       f"higher-ratio picks from {item['selected']} selected" for item in result))
    print(output)
