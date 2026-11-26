"""Regression tests for pair traces, bond replay, and held-out explanation selection."""

import numpy as np
import pytest
from rdkit import Chem
import torch
from torch_geometric.nn import global_add_pool

from core import dataset, explanation_benchmark as benchmark, explanations, learning, screening
from core.explanation_html import gate_html
from core.storage import write_json


class EdgeScore(torch.nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

    def forward(self, batch):
        weights = batch.edge_attr[:, 0] * self.multiplier
        return global_add_pool(weights, batch.batch[batch.edge_index[0]], size=batch.num_graphs)


@pytest.fixture
def frozen_fixture():
    smiles = {"parent": "[Na+]", "product": "CC=O", "long": "CCOC=O"}
    cache = {
        "graphs": {key: dataset.molecule_graph(Chem.MolFromSmiles(smi)) for key, smi in smiles.items()},
        "structures": smiles, "labels": {"bbb": {}, "pgp": {}},
        "relationships": [["parent", "product"]],
    }
    ensembles = {
        endpoint: ([EdgeScore(sign * factor) for factor in (0.8, 0.9, 1.0, 1.1, 1.2)], {
            "threshold": 0.55, "slope": 1, "intercept": 0, "seeds": learning.SEEDS,
        })
        for endpoint, sign in (("bbb", 1), ("pgp", -1))
    }
    return cache, ensembles, learning.device_for("cpu")


def test_documented_lookup_is_exact_and_directional(frozen_fixture):
    cache, _, _ = frozen_fixture
    assert explanations.documented_pair(cache, "parent", "product") == ("parent", "product")
    for pair in (("product", "parent"), ("Parent", "product"), ("parent", "prod"), ("parent", "long")):
        with pytest.raises(ValueError, match="exact"):
            explanations.documented_pair(cache, *pair)


def test_stale_cache_and_model_artifacts_are_rejected(monkeypatch, tmp_path):
    cache_path = tmp_path / "training.pt"
    dataset.save_torch(cache_path, {"data_id": "old"})
    monkeypatch.setattr(dataset, "CACHE", cache_path)
    monkeypatch.setattr(dataset, "audit_inputs", lambda: ({}, "new"))
    with pytest.raises(ValueError, match="graph cache is stale"):
        explanations.load_dataset(include_structures=True)

    monkeypatch.setattr(learning, "MODELS", tmp_path)
    path = learning.checkpoint_path("bbb", 11)
    calibration = {"data_id": "old", "endpoint": "bbb", "model_version": learning.MODEL_VERSION,
                   "seeds": [11], "checkpoint_sha256": {"11": "old_checkpoint"}}
    write_json(path.with_name("calibration.json"), calibration)
    with pytest.raises(ValueError, match="Stale bbb calibration"):
        explanations.verified_ensembles({"data_id": "new"}, torch.device("cpu"))

    state = learning.GINE().state_dict()
    dataset.save_torch(path, {"config": {**calibration, "seed": 11}, "best_weights": state, "complete": True})
    with pytest.raises(ValueError, match="checkpoints changed"):
        explanations.verified_ensembles({"data_id": "old"}, torch.device("cpu"))


def test_mask_preserves_features_and_removes_both_edges(frozen_fixture):
    cache, _, _ = frozen_fixture
    _, bonds = explanations.checked_structure(cache, "product")
    graph = dataset.graph_data(cache, "product", "bbb")
    masked = explanations.masked_graph(graph, bonds, (0,))
    assert masked.x is graph.x
    assert graph.edge_index.shape[1] == 4
    assert torch.equal(masked.edge_index, graph.edge_index[:, 2:])
    assert torch.equal(masked.edge_attr, graph.edge_attr[2:])
    wrong_order = (graph.x, graph.edge_index.flip(1), graph.edge_attr.flip(0))
    cache["graphs"]["product"] = wrong_order
    with pytest.raises(ValueError, match="mapping could not be verified"):
        explanations.checked_structure(cache, "product")


def test_decisions_replay_with_original_signs_and_bbb_skip(frozen_fixture, monkeypatch):
    cache, ensembles, device = frozen_fixture
    scores = screening.sequential_scores(cache, ["parent", "product"], ensembles, device)
    assert scores["parent"]["stage"] == "bbb_rejected"
    assert scores["product"]["stage"] == "pass"
    for endpoint, sign in (("bbb", 1), ("pgp", -1)):
        _, bonds = explanations.checked_structure(cache, "product")
        entry = explanations.explain_endpoint(cache, "product", endpoint, ensembles[endpoint],
                                               device, bonds, scores["product"])
        assert entry["probability"] == scores["product"][explanations.PROBABILITY_FIELDS[endpoint]]
        assert entry["decision"] == "pass"
        strongest = entry["bond_effects"][0]
        assert strongest["bond_type"] == "DOUBLE"
        assert np.sign(strongest["mean_effect"]) == sign
        assert strongest["directional_agreement"] == 1
        assert list(strongest["seed_effects"]) == list(map(str, learning.SEEDS))
        # The toy BBB seed 37 sums its two directed edge weights per remaining bond.
        calibration = ensembles[endpoint][1]
        expected = learning.calibrated_probability(sign * np.array([2, 2 / 3]), calibration)
        assert strongest["seed_effects"]["37"] == pytest.approx(expected[0] - expected[1])

    mol, bonds = explanations.checked_structure(cache, "parent")
    entry = explanations.explain_endpoint(cache, "parent", "bbb", ensembles["bbb"], device, bonds, scores["parent"])
    assert entry["bond_effects"] == []
    assert "No chemical bonds to mask" in gate_html(mol, entry, 0)
    monkeypatch.setattr(explanations, "single_bond_replay", lambda *_: pytest.fail("P-gp replay after BBB rejection"))
    skipped = explanations.explain_endpoint(cache, "parent", "pgp", ensembles["pgp"], device, bonds, scores["parent"])
    assert "P-gp skipped" in gate_html(mol, skipped, 0)
    assert explanations.support_evidence(scores["parent"]) == {"value": 0, "reason": "zero_after_rejection"}


def test_replay_rejects_equal_within_tolerance_but_different_gate():
    screening_score = {"bbb_probability": 0.49999999, "bbb_seed_sd": 0, "stage": "bbb_rejected"}
    with pytest.raises(ValueError, match="score and gate"):
        explanations.check_replay("bbb", np.zeros(5), {"slope": 1, "intercept": 0, "threshold": 0.5},
                                  screening_score)


def test_structure_reports_ambiguous_and_insufficient_mapping():
    mol = Chem.MolFromSmiles
    ambiguous = explanations.structural_alignment(mol("CC"), mol("CCC"))
    assert ambiguous["status"] == "unavailable"
    assert "ambiguous" in ambiguous["reason"]
    too_small = explanations.structural_alignment(mol("N"), mol("CNCO"))
    assert too_small["status"] == "unavailable"
    assert "less than half" in too_small["reason"]
    unique = explanations.structural_alignment(mol("NCCO"), mol("NCCOF"))
    assert unique["status"] == "available"
    assert unique["heavy_atom_coverage"] == {"parent": 1, "metabolite": 0.8}
    assert unique["unmatched"]["metabolite"] == {"atom_indices": [4], "bond_indices": [3]}


def test_cohort_is_deterministic_disjoint_and_reports_shortfalls():
    pairs = [(f"parent{i}", f"child{i}", f"group{i // 2}") for i in range(30)]
    groups = {key: group for parent, child, group in pairs for key in (parent, child)}
    groups.update({"bbb_train": "group0", "pgp_cal": "group1"})
    cache = {
        "groups": groups, "relationships": [p[:2] for p in pairs],
        "reserved_groups": [f"group{i}" for i in range(14)],
        "splits": {"bbb": {"train": ["bbb_train"], "validation": [], "calibration": []},
                   "pgp": {"train": [], "validation": [], "calibration": ["pgp_cal"]}},
    }
    held_out = benchmark.held_out_pairs(cache)
    assert {p[2] for p in held_out} == {f"group{i}" for i in range(2, 14)}
    stages = ["bbb_rejected", "pgp_flagged", "pass"]
    scores = {f"child{i}": {"stage": stages[i % 3]} for i in range(30)}
    cohort, counts = benchmark.select_cohort(held_out, scores)
    assert (cohort, counts) == benchmark.select_cohort(held_out[::-1], scores)
    assert len(cohort) == len({row["group_id"] for row in cohort})
    assert list(counts) == stages
    assert all(row["shortfall"] == 10 - row["selected"] for row in counts.values())
    assert sum(row["selected"] for row in counts.values()) == len(cohort) == 12


def test_benchmark_rotates_all_seeds_and_uses_group_paired_differences(frozen_fixture):
    cache, ensembles, device = frozen_fixture
    scores = screening.sequential_scores(cache, ["long"], ensembles, device)
    _, bonds = explanations.checked_structure(cache, "long")
    args = cache, "long", "bbb", ensembles["bbb"], device, bonds, scores["long"]
    case = benchmark.evaluate_molecule(*args)
    assert case == benchmark.evaluate_molecule(*args)
    assert case["mask_size"] == 2
    assert [row["held_out_seed"] for row in case["rotations"]] == list(learning.SEEDS)
    assert all(row["held_out_seed"] not in row["ranking_seeds"] for row in case["rotations"])
    assert all(3 in row["top_bond_indices"] and len(row["top_bond_indices"]) == 2 for row in case["rotations"])
    assert case["seed_top_bond_overlap"] == 1
    pair = {"group_id": "first", "molecules": {"parent": {"bbb": case}, "metabolite": {"bbb": case}}}
    summary = benchmark.summarize_endpoint([pair], "bbb")
    assert summary["eligible_groups"] == 1
    assert summary["paired_group_bootstrap_95_ci"] is None
    pair2 = {**pair, "group_id": "second", "molecules": {
        **pair["molecules"], "metabolite": {"bbb": {"skipped": "fewer_than_three_bonds"}},
    }}
    summary = benchmark.summarize_endpoint([pair, pair2], "bbb")
    assert (summary["eligible_groups"], summary["eligible_cases"], summary["skipped_cases"]) == (2, 3, 1)
    assert summary["paired_group_bootstrap_95_ci"] == pytest.approx([summary["effect_over_random"]] * 2)
