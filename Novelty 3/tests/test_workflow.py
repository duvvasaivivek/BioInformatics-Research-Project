"""Regression checks for stage gating and evaluation with frozen fixture weights."""

import csv
import json

import numpy as np
import pytest
from rdkit import Chem
import torch

from core import learning, screening, validation
from core.audit import IdentityStore
from core.dataset import molecule_graph, save_torch
from core.sources import digest
from core.storage import write_json


def test_sequential_gate_preserves_metabolite_after_parent_rejection(monkeypatch):
    def fixture_score(cache, endpoint, keys, ensemble, device):
        if endpoint == "bbb":
            assert keys == ["parent", "product"]
            return {"parent": (0.2, 0.01), "product": (0.8, 0.02)}
        assert keys == ["product"]
        return {"product": (0.3, 0.03)}

    monkeypatch.setattr(screening, "score", fixture_score)
    ensembles = {task: ([], {"threshold": 0.5}) for task in ("bbb", "pgp")}
    scores = screening.sequential_scores({}, ["parent", "product"], ensembles, None)
    assert scores["parent"]["stage"] == "bbb_rejected"
    assert scores["parent"]["pgp_substrate_probability"] is None
    assert scores["product"]["stage"] == "pass"
    assert scores["product"]["support"] == pytest.approx(0.67)


def test_review_with_frozen_checkpoints_and_missing_labels(monkeypatch, tmp_path):
    keys = ("parent", "unknown", "positive", "test")
    labels = {"bbb": {"parent": 0, "positive": 1, "test": 0},
              "pgp": {"test": 1}}
    cache = {
        "data_id": "fixture", "graphs": {key: molecule_graph(Chem.MolFromSmiles(smiles))
                                         for key, smiles in zip(keys, ("CCO", "CC", "CCN", "[Na+]"))},
        "labels": labels, "curated": {"bbb": [], "pgp": []},
        "splits": {task: {"test": ["test"]} for task in labels},
        "groups": {key: "test" if key == "test" else "reserved" for key in keys},
        "families": {key: "family" for key in keys if key != "test"},
        "reserved_groups": ["reserved"], "relationships": [["parent", "unknown"], ["parent", "positive"]],
    }
    monkeypatch.setattr(learning, "MODELS", tmp_path / "models")
    monkeypatch.setattr(screening, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(screening, "load_dataset", lambda: cache)
    model = learning.GINE()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    for endpoint in labels:
        path = learning.checkpoint_path(endpoint, 11)
        config = {"endpoint": endpoint, "data_id": "fixture", "seed": 11,
                  "model_version": learning.MODEL_VERSION}
        save_torch(path, {"config": config, "complete": True, "best_weights": model.state_dict()})
        write_json(path.with_name("calibration.json"), {
            **config, "seeds": [11], "slope": 1.0, "intercept": 0.0,
            "threshold": 0.4 if endpoint == "bbb" else 0.6,
            "checkpoint_sha256": {"11": digest(path.read_bytes())},
        })
    screening.run_pipeline("cpu", [2])
    results = tmp_path / "results"
    with (results / "screening.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["metabolite_stage"] for row in rows} == {"pass"}
    report = json.loads((results / "evaluation.json").read_text())
    assert report["endpoint_metrics"]["pgp"]["endpoint_test"]["positive"] == 1
    assert report["endpoint_metrics"]["bbb"]["held_out_metabolite_annotations"]["positive"] == 1
    policy = report["fixed_budget"]["policies"]["parent_relative"][0]
    assert policy["selected"] == 1  # one per family even with the larger budget
    assert policy["joint"]["unknown"] == 1
    assert policy["joint"]["negative"] == 0


def test_threshold_keeps_ties_and_meets_requested_sensitivity():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.8, 0.2, 0.8, 0.3])
    threshold = learning.sensitivity_threshold(y, p, 0.5)
    assert threshold == 0.8
    assert learning.metrics(y, p, threshold)["sensitivity"] == 0.75


def test_paired_validation_excludes_new_family_bridge_to_calibration():
    identities = IdentityStore()
    parent = identities.get("CCO")
    child = identities.get("CC=O")
    a, b = parent["inchikey"], child["inchikey"]
    compounds = {"parent": parent, "child": child}
    study = {"parent": "parent", "id": "study", "observations": [
        {"condition": "first", "ratio": {"parent": 2.0, "child": 1.0}},
        {"condition": "second", "ratio": {"parent": 2.0, "child": 4.0}},
    ]}
    rows = validation.measured_pairs([study], compounds)
    cache = {"groups": {a: a, b: b}, "relationships": [], "graphs": {},
             "splits": {"bbb": {"train": [], "validation": [], "calibration": [a]},
                        "pgp": {"train": [], "validation": [], "calibration": []}}}
    validation.holdout_groups(cache, rows, compounds, identities.molecules)
    assert rows[0]["model_development_overlap"] == ["bbb:calibration"]
    assert rows[0]["median_measured_fold_vs_parent"] == 1.25


def test_pgp_cascade_comparison_excludes_bbb_development_groups():
    from core.pgp_baseline import comparison_cohorts

    cache = {
        "labels": {"pgp": {"independent": 0, "overlapping": 1, "bbb_fail": 1, "child": 1}},
        "relationships": [["parent", "child"]],
        "splits": {"bbb": {"train": ["related"], "validation": [], "calibration": []},
                   "pgp": {"validation": [], "calibration": [],
                           "test": ["independent", "overlapping", "bbb_fail"]}},
        "groups": {"independent": "a", "overlapping": "b", "related": "b", "bbb_fail": "c", "child": "d"},
    }
    scores = {"independent": (0.7, 0), "overlapping": (0.8, 0), "bbb_fail": (0.4, 0), "child": (0.5, 0)}
    cohorts = comparison_cohorts(cache, scores, 0.5)
    assert cohorts["endpoint_test"] == ["independent", "overlapping", "bbb_fail"]
    assert cohorts["endpoint_test_after_bbb"] == ["independent"]
    assert cohorts["reserved_metabolite_annotations_after_bbb"] == ["child"]


def test_metabolite_cutoff_fit_only_scores_development_and_locks_artifacts(monkeypatch, tmp_path):
    from core import pgp_metabolites as pilot

    dev = [{"split": "development", "label": label, "metabolite_inchikey": f"dev{i:02}",
            "group_id": f"dev{i}", "exclusions": []} for i, label in enumerate([1] * 10 + [0, 0])]
    test = [{"split": "test", "label": label, "metabolite_inchikey": f"test{i}",
             "group_id": f"test{i}", "exclusions": []} for i, label in enumerate([1, 1] + [0] * 6)]
    seen = []

    def predictions(cache, records, rows, models):
        assert len({row["split"] for row in rows}) == 1
        seen.append(rows[0]["split"])
        y = np.array([row["label"] for row in rows])
        if seen[-1] == "development":
            p = np.array([0.1, 0.4, *([0.9] * 8), 0.2, 0.3])
            return rows, y, {"gnn": p, "xgboost": p}
        return rows, y, {"gnn": np.where(y == 1, 0.9, 0.7),
                         "xgboost": np.where(y == 1, 0.9, 0.2)}

    monkeypatch.setattr(pilot, "LOCK", tmp_path / "lock.json")
    monkeypatch.setattr(pilot, "REPORT", tmp_path / "report.json")
    monkeypatch.setattr(pilot, "cohort", lambda: ({}, {}, dev + test, {"dataset": "fixture"}))
    monkeypatch.setattr(pilot, "frozen_models", lambda *_: (None, None, (None, {"threshold": 0.6}), None, None))
    monkeypatch.setattr(pilot, "score_split", predictions)
    pilot.fit_cutoffs()
    assert seen == ["development"]
    assert json.loads(pilot.LOCK.read_text())["cutoffs"] == {"gnn": 0.4, "xgboost": 0.4}
    pilot.evaluate_cutoffs()
    assert seen == ["development", "test"]
    assert json.loads(pilot.REPORT.read_text())["decision"] == "xgboost_candidate"
    monkeypatch.setattr(pilot, "cohort", lambda: ({}, {}, dev + test, {"dataset": "changed"}))
    with pytest.raises(ValueError, match="changed"):
        pilot.evaluate_cutoffs()
    assert seen == ["development", "test"]


def test_metabolite_study_cannot_span_new_cohorts():
    from rdkit import Chem
    from core.pgp_metabolites import group_cases

    molecules = {"a": "CCC", "b": "CCO", "c": "CNC", "d": "CNCO"}
    records = {key: {"connectivity_key": key} for key in molecules}
    parsed = {key: Chem.MolFromSmiles(smiles) for key, smiles in molecules.items()}
    rows = [{"parent_inchikey": a, "metabolite_inchikey": b, "study": "same-study", "split": split,
             "exclusions": []} for a, b, split in (("a", "b", "development"), ("c", "d", "test"))]
    group_cases({"relationships": [], "splits": {}, "labels": {"pgp": {}}}, rows, [], records, parsed)
    assert rows[0]["group_id"] == rows[1]["group_id"]
    assert all("group_crosses_new_splits" in row["exclusions"] for row in rows)
