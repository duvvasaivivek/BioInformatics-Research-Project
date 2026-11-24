"""Sequential inference and retrospective evaluation at fixed assay budgets."""

from .audit import write_csv
from .dataset import ENDPOINTS, load_dataset, partition_summary
from .learning import calibrated_ensemble, device_for, metrics, score
from .storage import ROOT, write_json

RESULTS = ROOT / "results"


def sequential_scores(cache, keys, ensembles, device):
    bbb = score(cache, "bbb", keys, ensembles["bbb"], device)
    bbb_cutoff = ensembles["bbb"][1]["threshold"]
    pgp_cutoff = ensembles["pgp"][1]["threshold"]
    passed_bbb = [key for key in keys if bbb[key][0] >= bbb_cutoff]
    pgp = score(cache, "pgp", passed_bbb, ensembles["pgp"], device)
    scores = {}
    for key in keys:
        bbb_p, bbb_sd = bbb[key]
        pgp_p, pgp_sd = pgp.get(key, (None, None))
        stage = "bbb_rejected" if pgp_p is None else "pgp_flagged" if pgp_p >= pgp_cutoff else "pass"
        support = (max(0.0, min(bbb_p - bbb_sd, 1 - pgp_p - pgp_sd)) if stage == "pass" else 0.0)
        scores[key] = {"bbb_probability": bbb_p, "bbb_seed_sd": bbb_sd,
                       "pgp_substrate_probability": pgp_p, "pgp_seed_sd": pgp_sd,
                       "stage": stage, "support": support}
    return scores


def relationship_rows(cache, scores):
    reserved = set(cache["reserved_groups"])
    rows = []
    for parent, metabolite in cache["relationships"]:
        precursor = scores[parent]
        product = scores[metabolite]
        row = {"family_id": cache["families"][parent], "group_id": cache["groups"][parent],
               "reserved_group": cache["groups"][parent] in reserved,
               "parent_inchikey": parent, "metabolite_inchikey": metabolite,
               "parent_relative_support": (
                   product["support"] - precursor["support"] if product["stage"] == "pass" else None)}
        row.update({f"parent_{field}": value for field, value in precursor.items()})
        row.update({f"metabolite_{field}": value for field, value in product.items()})
        rows.append(row)
    return sorted(rows, key=lambda row: (
        row["parent_relative_support"] is None, -(row["parent_relative_support"] or 0),
        row["parent_inchikey"], row["metabolite_inchikey"]))


def run_screen(cache, ensembles, device):
    keys = sorted(cache["families"])
    scores = sequential_scores(cache, keys, ensembles, device)
    rows = relationship_rows(cache, scores)
    if not rows:
        raise ValueError("The prepared data contains no documented metabolic relationships.")
    path = RESULTS / "screening.csv"
    write_csv(path, rows, list(rows[0]))
    passed = sum(entry["stage"] == "pass" for entry in scores.values())
    print(f"Screened {len(keys)} distinct molecules and {len(rows)} relationships; {passed} molecules passed both stages.")
    print(path)
    return rows


def outcome(cache, endpoint, key):
    bbb, pgp = (cache["labels"][task].get(key) for task in ENDPOINTS)
    if endpoint == "bbb":
        return bbb
    if bbb is None or pgp is None:
        return None
    return int(bbb == 1 and pgp == 0)


def rank_candidates(rows, group_field="family_id"):
    policies = {
        "parent_relative": ("metabolite_stage", "parent_relative_support"),
        "independent_metabolite": ("metabolite_stage", "metabolite_support"),
        "parent_guided": ("parent_stage", "parent_support"),
    }
    selections = {}
    for policy, (stage, ranking) in policies.items():
        picked, families, seen = [], set(), set()
        ranked = sorted((row for row in rows if row[stage] == "pass"),
                        key=lambda row: (-row[ranking], row["parent_inchikey"], row["metabolite_inchikey"]))
        for row in ranked:
            key = row["metabolite_inchikey"]
            if key not in seen and row[group_field] not in families:
                families.add(row[group_field])
                seen.add(key)
                picked.append(row)
        selections[policy] = picked
    return selections


def fixed_budget(cache, rows, budgets):
    rows = [row for row in rows if row["reserved_group"]]
    candidates = {row["metabolite_inchikey"] for row in rows}
    positives = {
        endpoint: {key for key in candidates if outcome(cache, endpoint, key) == 1}
        for endpoint in ("bbb", "joint")
    }
    reports = {}
    for policy, ranked in rank_candidates(rows).items():
        picked = [row["metabolite_inchikey"] for row in ranked]
        reports[policy] = []
        for budget in budgets:
            selected = picked[:budget]
            results = {}
            for endpoint in positives:
                known = sum(outcome(cache, endpoint, key) is not None for key in selected)
                hits = len(set(selected) & positives[endpoint])
                results[endpoint] = {
                    "known": known, "unknown": len(selected) - known, "positive": hits,
                    "negative": known - hits,
                    "precision_among_known": hits / known if known else None,
                    "fraction_of_known_positives_recovered": (
                        hits / len(positives[endpoint]) if positives[endpoint] else None),
                }
            reports[policy].append({"budget": budget, "selected": len(selected), **results,
                                    "selected_metabolites": selected})
    return {"pool_metabolites": len(candidates),
            "pool_families": len({row["family_id"] for row in rows}),
            "pool_positive": {endpoint: len(keys) for endpoint, keys in positives.items()},
            "policies": reports}


def endpoint_report(cache, endpoint, ensemble, device):
    metabolites = {product for _, product in cache["relationships"]}
    labels = cache["labels"][endpoint]
    annotated = sorted(metabolites & labels.keys())
    curated = sorted(metabolites & set(cache["curated"][endpoint]))
    test = cache["splits"][endpoint]["test"]
    predictions = score(cache, endpoint, sorted(set(annotated) | set(test)), ensemble, device)
    threshold = ensemble[1]["threshold"]
    report = {}
    for name, keys in (("endpoint_test", test), ("held_out_metabolite_annotations", annotated),
                       ("curated_metabolite_measurements", curated)):
        report[name] = metrics([labels[key] for key in keys],
                               [predictions[key][0] for key in keys], threshold)
        report[name]["groups"] = len({cache["groups"][key] for key in keys})
    return report


def run_pipeline(choice, budgets=None):
    cache = load_dataset()
    device = device_for(choice)
    ensembles = {task: calibrated_ensemble(cache, task, device) for task in ENDPOINTS}
    rows = run_screen(cache, ensembles, device)
    if budgets is None:
        return
    labels = cache["labels"]["bbb"]
    candidates = [row for row in rows if row["reserved_group"]
                  and labels.get(row["parent_inchikey"]) == 0
                  and labels.get(row["metabolite_inchikey"]) == 1]
    report = {
        "data_id": cache["data_id"],
        "evidence": "Retrospective source annotations with separately reported curated measurements.",
        "support_definition": "For a two-stage pass, min(BBB probability minus seed SD, "
                              "P-gp nonsubstrate probability minus seed SD), floored at zero. "
                              "Other outcomes have zero ranking support. Support is a ranking heuristic.",
        "partitions": partition_summary(cache),
        "calibration": {task: ensemble[1] for task, ensemble in ensembles.items()},
        "endpoint_metrics": {task: endpoint_report(cache, task, ensemble, device)
                             for task, ensemble in ensembles.items()},
        "fixed_budget": fixed_budget(cache, rows, budgets),
        "candidate_bbb_annotation_reversals": [
            {field: row[field] for field in ("family_id", "parent_inchikey", "parent_stage",
                                            "metabolite_inchikey", "metabolite_stage")}
            for row in candidates
        ],
    }
    path = RESULTS / "evaluation.json"
    write_json(path, report)
    for task, results in report["endpoint_metrics"].items():
        print(f"[{task}] held-out metabolite annotations: {results['held_out_metabolite_annotations']}")
    print(path)
