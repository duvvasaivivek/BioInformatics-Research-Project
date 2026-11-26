"""Held-out, group-paired benchmark of frozen bond sensitivity rankings."""

from collections import Counter, defaultdict
from itertools import combinations
from math import ceil

import numpy as np

from .dataset import ENDPOINTS, load_dataset
from .explanations import (PROBABILITY_FIELDS, RESULTS, ROLES, STAGES, checked_structure,
                           provenance, replay, single_bond_replay, verified_ensembles)
from .learning import calibrated_probability, device_for
from .screening import sequential_scores
from .sources import digest
from .storage import atomic_bytes, write_json


DESIGN_SEED = 2026
PER_STAGE = 10
RANDOM_MASKS = 100
BOOTSTRAPS = 10000


def stable_hash(*parts):
    return digest("|".join(map(str, (DESIGN_SEED, *parts))).encode())


def held_out_pairs(cache):
    groups = cache["groups"]
    excluded = {groups[key] for endpoint in ENDPOINTS for split in ("train", "validation", "calibration")
                for key in cache["splits"][endpoint][split]}
    eligible = set(cache["reserved_groups"]) - excluded
    return sorted({(parent, metabolite, groups[parent]) for parent, metabolite in cache["relationships"]
                   if groups[parent] == groups[metabolite] and groups[parent] in eligible})


def select_cohort(pairs, scores):
    used_groups, selected = set(), []
    counts = {}
    ordered = sorted(pairs, key=lambda pair: (stable_hash(*pair[:2]), pair[:2]))
    for stage in STAGES:
        candidates = [pair for pair in ordered if scores[pair[1]]["stage"] == stage]
        chosen = 0
        for parent, metabolite, group in candidates:
            if group in used_groups:
                continue
            selected.append({"parent": parent, "metabolite": metabolite, "group_id": group,
                             "metabolite_stage": stage})
            used_groups.add(group)
            chosen += 1
            if chosen == PER_STAGE:
                break
        counts[stage] = {"candidate_pairs": len(candidates), "candidate_groups": len({p[2] for p in candidates}),
                         "selected": chosen, "shortfall": PER_STAGE - chosen}
    return selected, counts


def top_bonds(effects, count):
    return tuple(sorted(range(len(effects)), key=lambda i: (-abs(effects[i]), i))[:count])


def evaluate_molecule(cache, key, endpoint, ensemble, device, bonds, screening):
    if screening[PROBABILITY_FIELDS[endpoint]] is None:
        return {"skipped": "bbb_rejected_pgp_skipped"}
    graph, original, effects = single_bond_replay(cache, key, endpoint, ensemble, device, bonds, screening)
    if len(bonds) < 3:
        return {"skipped": "fewer_than_three_bonds", "bond_count": len(bonds)}

    models, calibration = ensemble
    seeds = calibration["seeds"]
    size = min(len(bonds) - 1, max(2, ceil(len(bonds) / 5)))
    individual_top = [set(top_bonds(row, size)) for row in effects]
    overlap = np.mean([len(a & b) / size for a, b in combinations(individual_top, 2)])
    rotations = []
    for i, (seed, model) in enumerate(zip(seeds, models)):
        mean_effect = np.delete(effects, i, axis=0).mean(axis=0)
        ranking = top_bonds(mean_effect, size)
        top_mask = tuple(sorted(ranking))
        rng = np.random.default_rng(int(stable_hash(key, endpoint, seed)[:16], 16))
        random_masks = [tuple(sorted(map(int, rng.choice(len(bonds), size, replace=False))))
                        for _ in range(RANDOM_MASKS)]
        masks = list(dict.fromkeys([(), top_mask, *random_masks]))
        logits = replay(model, graph, bonds, masks, device)
        probabilities = calibrated_probability(logits, calibration)
        if not np.isclose(probabilities[0], original[i], rtol=1e-5, atol=1e-7):
            raise ValueError(f"Unmasked {endpoint} joint replay changed the score for seed {seed}.")
        changes = dict(zip(masks, np.abs(probabilities[0] - probabilities)))
        top = float(changes[top_mask])
        random = float(np.mean([changes[mask] for mask in random_masks]))
        direction = np.mean(np.sign(mean_effect[list(ranking)]) == np.sign(effects[i, list(ranking)]))
        rotations.append({
            "held_out_seed": seed, "ranking_seeds": [other for other in seeds if other != seed],
            "top_bond_indices": list(ranking), "top_abs_change": top, "random_mean_abs_change": random,
            "effect_over_random": top - random, "directional_agreement": float(direction),
        })
    return {"bond_count": len(bonds), "mask_size": size, "seed_top_bond_overlap": float(overlap),
            "rotations": rotations}


def summarize_endpoint(pairs, endpoint):
    by_group, skips = defaultdict(list), Counter()
    for pair in pairs:
        for role in ROLES:
            case = pair["molecules"][role][endpoint]
            if "skipped" in case:
                skips[case["skipped"]] += 1
            else:
                by_group[pair["group_id"]].append(case)

    fields = ("top_abs_change", "random_mean_abs_change", "effect_over_random", "directional_agreement")
    group_means = {}
    for group, cases in sorted(by_group.items()):
        rotations = [rotation for case in cases for rotation in case["rotations"]]
        group_means[group] = {field: float(np.mean([rotation[field] for rotation in rotations])) for field in fields}
        group_means[group]["seed_top_bond_overlap"] = float(np.mean([case["seed_top_bond_overlap"] for case in cases]))

    summary = {field: (float(np.mean([row[field] for row in group_means.values()])) if group_means else None)
               for field in (*fields, "seed_top_bond_overlap")}
    interval = None
    if len(group_means) >= 2:
        values = np.array([row["effect_over_random"] for row in group_means.values()])
        rng = np.random.default_rng(int(stable_hash(endpoint, "bootstrap")[:16], 16))
        samples = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True)
        interval = np.quantile(samples.mean(axis=1), [0.025, 0.975]).tolist()
    return {
        "eligible_groups": len(group_means), "eligible_cases": sum(map(len, by_group.values())),
        "skipped_cases": sum(skips.values()), "skipped_by_reason": dict(skips), **summary,
        "paired_group_bootstrap_95_ci": interval, "group_means": group_means,
    }


def run_benchmark(choice):
    from .explanation_html import benchmark_html

    cache = load_dataset(include_structures=True)
    device = device_for(choice)
    ensembles = verified_ensembles(cache, device)
    pairs = held_out_pairs(cache)
    scores = sequential_scores(cache, sorted({key for pair in pairs for key in pair[:2]}), ensembles, device)
    cohort, counts = select_cohort(pairs, scores)
    print(f"Selected {len(cohort)} pairs from {len({pair[2] for pair in pairs})} held-out groups.")
    for i, pair in enumerate(cohort, 1):
        print(f"[{i}/{len(cohort)}] {pair['metabolite_stage']} · {pair['parent']} / {pair['metabolite']}")
        pair["molecules"] = {}
        for role in ROLES:
            key = pair[role]
            _, bonds = checked_structure(cache, key)
            pair["molecules"][role] = {
                endpoint: evaluate_molecule(cache, key, endpoint, ensembles[endpoint], device, bonds, scores[key])
                for endpoint in ENDPOINTS
            }
    report = {
        "provenance": provenance(cache, ensembles, device),
        "design": {"seed": DESIGN_SEED, "stage_order": list(STAGES), "max_per_stage": PER_STAGE,
                   "random_masks_per_rotation": RANDOM_MASKS, "bootstrap_resamples": BOOTSTRAPS},
        "pool": {"pairs": len(pairs), "groups": len({pair[2] for pair in pairs})},
        "cohort_counts": counts, "cohort": cohort,
        "endpoints": {endpoint: summarize_endpoint(cohort, endpoint) for endpoint in ENDPOINTS},
    }
    html = benchmark_html(report)
    write_json(RESULTS / "benchmark.json", report)
    atomic_bytes(RESULTS / "benchmark.html", html.encode("utf-8"))
    print(RESULTS / "benchmark.html")
    return report
