"""Regression checks for source integrity and evidence handling; no model fits."""

import csv
import json

import pytest

from core import audit
from core.sources import cached_source, digest


def test_directional_smiles_in_malformed_source_json_are_preserved(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(r'[{"SMILES": "C/C=C\C"}]', encoding="utf-8")
    records, repairs = audit.load_relationship_json(path)
    assert records[0]["SMILES"] == r"C/C=C\C"
    assert repairs == 1
    assert r"\C" in path.read_text(encoding="utf-8")


def test_unrelated_json_errors_are_fatal(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('[{"SMILES": "CCO",}]', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid source JSON"):
        audit.load_relationship_json(path)


def test_source_cache_detects_content_and_receipt_corruption(tmp_path):
    path = tmp_path / "raw.csv"
    receipt = tmp_path / "receipt.json"
    path.write_bytes(b"original")
    receipt.write_text(json.dumps({"url": "https://example.org/source", "sha256": digest(b"original")}))
    assert cached_source(path, receipt, "https://example.org/source")
    path.write_bytes(b"modified")
    assert cached_source(path, receipt, "https://example.org/source") is None
    receipt.write_text("not json")
    assert cached_source(path, receipt, "https://example.org/source") is None


def test_strict_identity_preserves_stereochemistry_and_salts():
    identities = audit.IdentityStore()
    first = identities.get("C[C@H](O)C(=O)O")
    second = identities.get("C[C@@H](O)C(=O)O")
    assert first["inchikey"] != second["inchikey"]
    assert first["connectivity_key"] == second["connectivity_key"]
    assert identities.get("CC(=O)[O-].[Na+]")["fragment_count"] == 2
    assert identities.get("NOT_A_SMILES") is None


def test_conflicting_labels_are_excluded_and_missing_labels_stay_unknown():
    records = [
        {"endpoint": "bbb", "inchikey": "a", "label": 0},
        {"endpoint": "bbb", "inchikey": "a", "label": 1},
        {"endpoint": "pgp", "inchikey": "b", "label": 1},
    ]
    labels, conflicts = audit.endpoint_labels(records)
    assert "a" not in labels["bbb"]
    assert conflicts["bbb"] == ["a"]
    assert "b" not in labels["bbb"]
    assert labels["pgp"]["b"] == 1


def test_shared_descendants_and_cycles_form_one_family():
    edges = [
        {"parent_inchikey": "a", "metabolite_inchikey": "c"},
        {"parent_inchikey": "b", "metabolite_inchikey": "c"},
        {"parent_inchikey": "c", "metabolite_inchikey": "a"},
        {"parent_inchikey": "d", "metabolite_inchikey": "e"},
    ]
    families = audit.family_components(edges)
    assert families["a"] == families["b"] == families["c"]
    assert families["d"] == families["e"] != families["a"]
    assert audit.family_components(list(reversed(edges))) == families
    assert list(families) == ["a", "b", "c", "d", "e"]


def test_nested_literature_schema_retains_intermediate_steps_and_citations():
    data = [{"drugName": "example", "smiles": "CCO", "doi": "10.example/paper",
             "metabolites": [{"smiles": "CC=O", "generation": 1, "metaboliteName": "M1",
                              "metabolites": [{"smiles": "CC(=O)O", "generation": 2}]}]}]
    edges = list(audit.source_relationships("metabolism_test", data))
    assert len(edges) == 2
    assert edges[0][1]["SMILES"] == "CCO"
    assert edges[1][1]["SMILES"] == "CC=O"
    assert edges[1][2]["SMILES"] == "CC(=O)O"
    assert edges[1][3]["source_doi"] == "10.example/paper"


def test_relationship_import_deduplicates_sources_and_excludes_self_edges(tmp_path, monkeypatch):
    reference = tmp_path / "reference.json"
    literature = tmp_path / "literature.json"
    reference.write_text(json.dumps([
        {"Parent molecule": {"SMILES": "CCO"}, "Metabolites": [
            {"SMILES": "CCO"}, {"SMILES": "CC=O"}, {"SMILES": "invalid"}]}
    ]), encoding="utf-8")
    literature.write_text(json.dumps([
        {"drugName": "example", "smiles": "CCO", "doi": "10.example/paper",
         "metabolites": [{"smiles": "CC=O"}]}
    ]), encoding="utf-8")
    monkeypatch.setattr(audit, "source_path", lambda name: reference if name == "metabolism_reference" else literature)
    edges, counts = audit.load_relationships(audit.IdentityStore())
    assert len(edges) == 1
    assert len(edges[0]["evidence"]) == 2
    assert counts["metabolism_reference"]["self_relationships_excluded"] == 1
    assert counts["metabolism_reference"]["invalid_metabolite_structures"] == 1
    assert counts["metabolism_test"]["duplicate_relationships"] == 1


def test_endpoint_adapter_counts_bad_labels_and_structures(tmp_path, monkeypatch):
    path = tmp_path / "bbb.tsv"
    path.write_text("SMILES\tBBB+/BBB-\nCCO\tBBB+\nCC=O\tunknown\ninvalid\tBBB-\n", encoding="utf-8")
    monkeypatch.setattr(audit, "source_path", lambda source: path)
    records, counts = audit.load_endpoint("bbb", audit.IdentityStore())
    assert len(records) == 1
    assert counts == {"raw_rows": 3, "invalid_labels": 1, "invalid_structures": 1}
    assert records[0]["label"] == 1


def curated_file(tmp_path, **overrides):
    row = {
        "record_id": "paper1_compound1", "smiles": "CCO", "endpoint": "bbb", "label": "1",
        "measurement_type": "logbb", "value": "0.2", "units": "log10 ratio", "species": "rat",
        "assay": "in vivo brain/plasma", "source_url": "doi:10.example/paper",
        "label_definition": "Positive when logBB >= -1 in this study", "notes": "",
    }
    row.update(overrides)
    path = tmp_path / "curated.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=audit.CURATION_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return path


@pytest.mark.parametrize("overrides", [
    {"measurement_type": "prediction"},
    {"endpoint": "pgp", "measurement_type": "inhibition"},
    {"label": ""},
    {"source_url": ""},
    {"value": "nan"},
    {"species": ""},
])
def test_invalid_curation_cannot_create_experimental_evidence(tmp_path, overrides):
    with pytest.raises(ValueError):
        audit.load_curated(curated_file(tmp_path, **overrides), audit.IdentityStore())


def test_valid_curation_retains_measurement_provenance(tmp_path):
    rows = audit.load_curated(curated_file(tmp_path), audit.IdentityStore())
    assert rows[0]["raw_value"] == "0.2"
    assert rows[0]["species"] == "rat"
    assert rows[0]["evidence_level"] == "manually_curated_experiment"


def test_empty_curation_is_valid_and_adds_no_labels(tmp_path):
    path = tmp_path / "curated.csv"
    path.write_text(",".join(audit.CURATION_FIELDS) + "\n", encoding="utf-8")
    assert audit.load_curated(path, audit.IdentityStore()) == []


def test_truncated_curation_row_fails_with_actionable_error(tmp_path):
    path = tmp_path / "curated.csv"
    path.write_text(",".join(audit.CURATION_FIELDS) + "\nrecord,CCO,bbb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different number of fields"):
        audit.load_curated(path, audit.IdentityStore())


def test_summary_does_not_turn_unknown_parent_into_negative():
    identities = audit.IdentityStore()
    parent = identities.get("CCO")["inchikey"]
    product = identities.get("CC=O")["inchikey"]
    records = [{"record_id": "bbb:2", "inchikey": product, "endpoint": "bbb", "label": 1,
                "source_id": "bbb", "evidence_level": "source_annotation"}]
    edges = [{"parent_inchikey": parent, "metabolite_inchikey": product,
              "evidence": [{"source_id": "metabolism_test"}]}]
    report, coverage, families = audit.summarize(records, edges, identities, {})
    assert report["annotation_intersections"]["parent_bbb_negative_metabolite_positive_pairs"] == 0
    assert coverage[0]["parent_bbb_label"] == ""
    assert coverage[0]["metabolite_pgp_substrate_label"] == ""
    assert report["endpoint_coverage"]["bbb"]["positive_metabolites"] == 1
    assert not report["training_authorized_by_audit"]
    assert report["status"] == "REVIEW_REQUIRED"


def test_audit_writes_complete_outputs_and_returns_review_exit_code(tmp_path, monkeypatch):
    identities = audit.IdentityStore()
    parent = identities.get("CCO")["inchikey"]
    product = identities.get("CC=O")["inchikey"]
    curation = tmp_path / "data/input/metabolite_measurements.csv"
    curation.parent.mkdir(parents=True)
    curation.write_text(",".join(audit.CURATION_FIELDS) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "verified_sources", lambda: [{"id": "fixture"}])
    monkeypatch.setattr(audit, "IdentityStore", lambda path: identities)
    def endpoint(source, identity_store):
        if source == "pgp":
            return [], {"raw_rows": 0}
        return [{"record_id": "bbb:2", "inchikey": product, "endpoint": "bbb", "label": 1,
                 "source_id": "bbb", "evidence_level": "source_annotation"}], {"raw_rows": 1}
    monkeypatch.setattr(audit, "load_endpoint", endpoint)
    monkeypatch.setattr(audit, "load_relationships", lambda store: ([
        {"parent_inchikey": parent, "metabolite_inchikey": product,
         "evidence": [{"source_id": "fixture"}]}], {}))
    assert audit.run_audit() == 2
    report = json.loads((tmp_path / "results/phase1/feasibility_report.json").read_text())
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["curation_sha256"] == digest(curation.read_bytes())
    assert (tmp_path / "data/processed/relationships.json").exists()
    assert (tmp_path / "results/phase1/candidate_coverage.csv").exists()
    assert not (tmp_path / "models").exists()
    completion = json.loads((tmp_path / "results/phase1/audit_run.json").read_text())
    assert completion["status"] == "REVIEW_REQUIRED"
    assert len(completion["artifact_sha256"]) == 8

    family_path = tmp_path / "data/processed/family_components.json"
    families = json.loads(family_path.read_bytes())
    legacy_bytes = json.dumps(dict(reversed(list(families.items()))), indent=2).encode("utf-8")
    family_path.write_bytes(legacy_bytes)
    assert audit.run_audit() == 2
    assert family_path.read_bytes() == legacy_bytes
    completion = json.loads((tmp_path / "results/phase1/audit_run.json").read_text())
    assert completion["artifact_sha256"]["data/processed/family_components.json"] == digest(legacy_bytes)

    family_path.write_text("{}", encoding="utf-8")
    assert audit.run_audit() == 2
    assert json.loads(family_path.read_bytes()) == families


def test_failed_audit_marks_previous_reports_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    def fail():
        raise ValueError("source verification failed")
    monkeypatch.setattr(audit, "perform_audit", fail)
    with pytest.raises(ValueError, match="source verification"):
        audit.run_audit()
    completion = json.loads((tmp_path / "results/phase1/audit_run.json").read_text())
    assert completion["status"] == "INCOMPLETE"
    assert "source verification failed" in completion["error"]
