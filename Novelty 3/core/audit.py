"""Evidence audit without model fitting or predicted biological labels."""

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import inchi
from rdkit.Chem.Scaffolds import MurckoScaffold

from .sources import SOURCE_SPECS, digest, source_path, verified_sources
from .storage import ROOT, atomic_bytes, write_json

CURATION_FIELDS = (
    "record_id", "smiles", "endpoint", "label", "measurement_type", "value",
    "units", "species", "assay", "source_url", "label_definition", "notes",
)
IDENTITY_POLICY = "strict-full-InChIKey-stereochemistry-preserved-no-salt-stripping-v1"


def write_csv(path, records, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    atomic_bytes(path, output.getvalue().encode("utf-8"))


def load_relationship_json(path):
    """Repair only invalid JSON backslash escapes, leaving the source unchanged.

    The pinned GLORYx test JSON contains unescaped SMILES bond directions such
    as '\\C'. Preserve those literal backslashes in the parsed structure and
    disclose the repair count. Other syntax errors remain fatal.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    repairs = 0
    try:
        values = json.loads(text)
    except json.JSONDecodeError as error:
        if error.msg != "Invalid \\escape":
            raise ValueError(f"Invalid source JSON in {Path(path).name}: {error}") from error
        text, repairs = re.subn(r'\\(?!["\\/bfnrtu])', lambda match: "\\\\", text)
        try:
            values = json.loads(text)
        except json.JSONDecodeError as repaired_error:
            raise ValueError(f"Unresolved source JSON error in {Path(path).name}: {repaired_error}") from repaired_error
    if not isinstance(values, list):
        raise ValueError(f"Expected a list of families in {Path(path).name}")
    return values, repairs


class IdentityStore:
    """Cache costly identities by supplied SMILES and explicit chemistry policy."""

    def __init__(self, path=None):
        self.path = path
        self.values = {}
        self.molecules = {}
        self.invalid = {}
        if path is not None and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("policy") == IDENTITY_POLICY and saved.get("rdkit") == rdBase.rdkitVersion:
                self.values = saved.get("values", {})

    def get(self, smiles):
        smiles = str(smiles).strip()
        if smiles not in self.values:
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            record = None
            if molecule is not None and molecule.GetNumAtoms():
                try:
                    key = inchi.MolToInchiKey(molecule, options="/W1")
                    if key:
                        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
                        record = {
                            "inchikey": key,
                            "connectivity_key": key.split("-")[0],
                            "canonical_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
                            "scaffold": scaffold,
                            "fragment_count": len(Chem.GetMolFrags(molecule)),
                        }
                except (ValueError, RuntimeError):
                    record = None
            self.values[smiles] = record
            if len(self.values) % 250 == 0:
                self.save()
        record = self.values[smiles]
        if record is None:
            self.invalid[smiles] = "Invalid structure or unavailable InChI identity"
            return None
        self.molecules[record["inchikey"]] = record
        return record

    def save(self):
        if self.path is not None:
            write_json(self.path, {"policy": IDENTITY_POLICY, "rdkit": rdBase.rdkitVersion,
                                   "values": self.values})


def load_endpoint(source_id, identities):
    path = source_path(source_id)
    spec = next(spec for spec in SOURCE_SPECS if spec[0] == source_id)
    source_url = f"https://raw.githubusercontent.com/{spec[1]}/{spec[2]}/{spec[3]}"
    delimiter = "\t" if source_id == "bbb" else ","
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        fields = reader.fieldnames or []
        if source_id == "bbb":
            label_column = next((name for name in ("BBB+/BBB-", "BBB+") if name in fields), None)
            required = {"SMILES", label_column}
        else:
            label_column = "bioclass"
            required = {"SMILES", label_column}
        if not required.issubset(fields):
            raise ValueError(f"Unexpected {source_id} schema: {fields}")
        records = []
        counts = Counter()
        for number, row in enumerate(reader, start=2):
            counts["raw_rows"] += 1
            if any(not isinstance(row.get(field), str) for field in required):
                raise ValueError(f"Incomplete {source_id} source row {number}")
            raw_label = row[label_column].strip()
            mapping = {"BBB+": 1, "BBB-": 0} if source_id == "bbb" else {"0": 0, "1": 1}
            if raw_label not in mapping:
                counts["invalid_labels"] += 1
                continue
            identity = identities.get(row["SMILES"])
            if identity is None:
                counts["invalid_structures"] += 1
                continue
            records.append({
                "record_id": f"{source_id}:{number}", "inchikey": identity["inchikey"],
                "endpoint": source_id, "label": mapping[raw_label],
                "measurement_type": "BBB annotation" if source_id == "bbb" else "P-gp substrate classification",
                "raw_value": row.get("logBB", ""), "units": "log10 ratio" if row.get("logBB", "") else "",
                "species": "not resolved per record", "assay": "not resolved per record",
                "reference": row.get("reference", ""), "reliability_group": row.get("group", ""),
                "source_id": source_id, "source_url": f"{source_url}#L{number}",
                "label_definition": "source BBB annotation" if source_id == "bbb" else "1=substrate; 0=nonsubstrate",
                "evidence_level": "source_annotation",
            })
        return records, dict(counts)


def load_curated(path, identities):
    if not path.exists():
        raise ValueError(f"Missing curation template: {path}")
    records = []
    ids = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(CURATION_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Curation file columns do not match the documented measurement template")
        for number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Curation row {number} has a different number of fields from the header")
            if not any(str(value or "").strip() for value in row.values()):
                continue
            required = ("record_id", "smiles", "endpoint", "label", "measurement_type",
                        "species", "assay", "source_url", "label_definition")
            if any(not row[name].strip() for name in required):
                raise ValueError(f"Curation row {number} is missing required experimental provenance")
            if row["record_id"] in ids:
                raise ValueError(f"Duplicate curated record ID on row {number}")
            ids.add(row["record_id"])
            if row["endpoint"] not in ("bbb", "pgp") or row["label"] not in ("0", "1"):
                raise ValueError(f"Curation row {number} requires endpoint bbb/pgp and an explicit 0/1 label")
            allowed = {"bbb": {"bbb_annotation", "logbb", "kp_uu_brain"},
                       "pgp": {"substrate_assay", "efflux_ratio"}}
            if row["measurement_type"] not in allowed[row["endpoint"]]:
                raise ValueError(f"Unsupported measurement type in curation row {number}; inhibition and predictions are ineligible")
            if not row["source_url"].startswith(("https://", "http://", "doi:")):
                raise ValueError(f"Curation row {number} requires a traceable publication URL or DOI")
            if row["measurement_type"] in ("logbb", "kp_uu_brain", "efflux_ratio"):
                try:
                    value = float(row["value"])
                except ValueError as error:
                    raise ValueError(f"Curation row {number} requires a numerical experimental value") from error
                if not math.isfinite(value) or not row["units"].strip():
                    raise ValueError(f"Curation row {number} requires finite values and units")
                if row["measurement_type"] != "logbb" and value < 0:
                    raise ValueError(f"Negative concentration/efflux ratio on curation row {number}")
            identity = identities.get(row["smiles"])
            if identity is None:
                raise ValueError(f"Invalid curated structure on row {number}")
            records.append({
                "record_id": "curated:" + row["record_id"], "inchikey": identity["inchikey"],
                "endpoint": row["endpoint"], "label": int(row["label"]),
                "measurement_type": row["measurement_type"], "raw_value": row["value"],
                "units": row["units"], "species": row["species"], "assay": row["assay"],
                "reference": row["source_url"], "reliability_group": "",
                "source_id": "curated", "source_url": row["source_url"],
                "label_definition": row["label_definition"], "evidence_level": "manually_curated_experiment",
            })
    return records


def source_relationships(source_id, records):
    """Read the flat reference schema and the nested literature-test schema."""
    for number, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Unexpected family record in {source_id} record {number}")
        if source_id == "metabolism_reference":
            parent = item.get("Parent molecule")
            products = item.get("Metabolites")
            if not isinstance(parent, dict) or not isinstance(products, list):
                raise ValueError(f"Unexpected family schema in {source_id} record {number}")
            for product in products:
                if not isinstance(product, dict):
                    raise ValueError(f"Unexpected metabolite record in {source_id} record {number}")
                yield number, parent, product, {"source_doi": "", "root_drug_name": ""}
            continue
        if not isinstance(item.get("smiles"), str) or not isinstance(item.get("metabolites"), list):
            raise ValueError(f"Unexpected family schema in {source_id} record {number}")
        pending = [item]
        while pending:
            node = pending.pop()
            children = node.get("metabolites", [])
            if not isinstance(children, list):
                raise ValueError(f"Unexpected nested metabolites in {source_id} record {number}")
            for child in children:
                if not isinstance(child, dict) or not isinstance(child.get("smiles"), str):
                    raise ValueError(f"Missing nested metabolite structure in {source_id} record {number}")
                parent = {"SMILES": node["smiles"], "name": node.get("drugName", node.get("metaboliteName", ""))}
                product = {"SMILES": child["smiles"], "name": child.get("metaboliteName", ""),
                           "generation": child.get("generation", "not supplied")}
                yield number, parent, product, {"source_doi": item.get("doi", "").strip(),
                                               "root_drug_name": item.get("drugName", "")}
                pending.append(child)


def load_relationships(identities):
    edges = {}
    statistics = {}
    for source_id in ("metabolism_reference", "metabolism_test"):
        records, repairs = load_relationship_json(source_path(source_id))
        counts = Counter(raw_parent_records=len(records), invalid_json_escapes_repaired=repairs)
        for number, parent_info, product_info, context in source_relationships(source_id, records):
            counts["raw_relationships"] += 1
            parent = identities.get(parent_info.get("SMILES", ""))
            if parent is None:
                counts["relationships_with_invalid_parent"] += 1
                continue
            product = identities.get(product_info.get("SMILES", ""))
            if product is None:
                counts["invalid_metabolite_structures"] += 1
                continue
            if parent["inchikey"] == product["inchikey"]:
                counts["self_relationships_excluded"] += 1
                continue
            pair = (parent["inchikey"], product["inchikey"])
            evidence = {
                "source_id": source_id, "source_record": number,
                "parent_identifiers": {key: value for key, value in parent_info.items() if key != "SMILES"},
                "metabolite_identifiers": {key: value for key, value in product_info.items() if key != "SMILES"},
                **context,
            }
            if pair in edges:
                counts["duplicate_relationships"] += 1
                edges[pair]["evidence"].append(evidence)
            else:
                edges[pair] = {
                    "parent_inchikey": pair[0], "metabolite_inchikey": pair[1],
                    "phase": product_info.get("Phase", "not supplied"),
                    "evidence_level": "documented_relationship_in_source_dataset",
                    "human_evidence": "source-level human metabolism; individual-study verification pending",
                    "evidence": [evidence],
                }
        statistics[source_id] = dict(counts)
    return list(edges.values()), statistics


def family_components(edges):
    """Undirected components join shared descendants and cycles without recursion."""
    adjacency = defaultdict(set)
    for edge in edges:
        parent, product = edge["parent_inchikey"], edge["metabolite_inchikey"]
        adjacency[parent].add(product)
        adjacency[product].add(parent)
    assignments = {}
    for start in sorted(adjacency):
        if start in assignments:
            continue
        pending = [start]
        members = set()
        while pending:
            node = pending.pop()
            if node in members:
                continue
            members.add(node)
            pending.extend(adjacency[node] - members)
        family_id = "family_" + hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:16]
        assignments.update({node: family_id for node in sorted(members)})
    return assignments


def endpoint_labels(records):
    labels = {"bbb": defaultdict(set), "pgp": defaultdict(set)}
    for record in records:
        labels[record["endpoint"]][record["inchikey"]].add(record["label"])
    usable = {task: {key: next(iter(values)) for key, values in rows.items() if len(values) == 1}
              for task, rows in labels.items()}
    conflicts = {task: sorted(key for key, values in rows.items() if len(values) > 1)
                 for task, rows in labels.items()}
    return usable, conflicts


def summarize(records, edges, identities, source_statistics):
    labels, conflicts = endpoint_labels(records)
    families = family_components(edges)
    metabolites = {edge["metabolite_inchikey"] for edge in edges}
    parents = {edge["parent_inchikey"] for edge in edges}
    outcomes = {}
    for task in ("bbb", "pgp"):
        labelled_metabolites = metabolites & labels[task].keys()
        positive = {key for key in labelled_metabolites if labels[task][key] == 1}
        negative = labelled_metabolites - positive
        curated_keys = {r["inchikey"] for r in records if r["endpoint"] == task and r["evidence_level"] == "manually_curated_experiment"}
        outcomes[task] = {
            "raw_measurement_rows": sum(r["endpoint"] == task for r in records),
            "repeated_identity_label_rows": sum(r["endpoint"] == task for r in records)
                                           - len({(r["inchikey"], r["label"]) for r in records if r["endpoint"] == task}),
            "usable_unique_molecules": len(labels[task]),
            "positive_molecules": sum(labels[task].values()),
            "negative_molecules": len(labels[task]) - sum(labels[task].values()),
            "conflicting_molecule_count": len(conflicts[task]),
            "conflicting_molecules": conflicts[task],
            "labelled_metabolites": len(labelled_metabolites),
            "positive_metabolites": len(positive), "negative_metabolites": len(negative),
            "independent_components_with_labelled_metabolites": len({families[key] for key in labelled_metabolites}),
            "components_with_positive_metabolites": len({families[key] for key in positive}),
            "manually_curated_labelled_metabolites": len(labelled_metabolites & curated_keys),
            "unique_source_groups": sorted({r["source_id"] for r in records if r["endpoint"] == task}),
        }
    joint = metabolites & labels["bbb"].keys() & labels["pgp"].keys()
    candidate_pairs = [edge for edge in edges
                       if labels["bbb"].get(edge["parent_inchikey"]) == 0
                       and labels["bbb"].get(edge["metabolite_inchikey"]) == 1]
    component_sizes = Counter(families.values())
    connectivity = defaultdict(set)
    scaffolds = defaultdict(set)
    for key, molecule in identities.molecules.items():
        connectivity[molecule["connectivity_key"]].add(key)
        if key in families and molecule["scaffold"]:
            scaffolds[molecule["scaffold"]].add(families[key])
    pair_rows = []
    for edge in edges:
        parent, product = edge["parent_inchikey"], edge["metabolite_inchikey"]
        pair_rows.append({
            "family_id": families[parent], "parent_inchikey": parent, "metabolite_inchikey": product,
            "parent_smiles": identities.molecules[parent]["canonical_smiles"],
            "metabolite_smiles": identities.molecules[product]["canonical_smiles"],
            "parent_bbb_label": labels["bbb"].get(parent, ""),
            "metabolite_bbb_label": labels["bbb"].get(product, ""),
            "parent_pgp_substrate_label": labels["pgp"].get(parent, ""),
            "metabolite_pgp_substrate_label": labels["pgp"].get(product, ""),
            "source_ids": "|".join(sorted({evidence["source_id"] for evidence in edge["evidence"]})),
            "metabolism_references": "|".join(sorted({evidence.get("source_doi", "") for evidence in edge["evidence"]} - {""})),
            "evidence_status": "candidate annotation match; publication/assay review required",
        })
    report = {
        "phase": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "identity_policy": IDENTITY_POLICY, "rdkit": rdBase.rdkitVersion,
        "status": "REVIEW_REQUIRED", "training_authorized_by_audit": False,
        "source_statistics": source_statistics, "endpoint_coverage": outcomes,
        "metabolism": {
            "unique_relationships": len(edges), "unique_precursors_including_intermediates": len(parents),
            "unique_metabolite_molecules": len(metabolites),
            "independent_family_components": len(component_sizes),
            "largest_component_molecules": max(component_sizes.values(), default=0),
            "metabolites_shared_by_multiple_parents": sum(count > 1 for count in Counter(e["metabolite_inchikey"] for e in edges).values()),
            "nonempty_scaffolds_spanning_family_components": sum(len(values) > 1 for values in scaffolds.values()),
        },
        "annotation_intersections": {
            "bbb_and_pgp_molecules": len(labels["bbb"].keys() & labels["pgp"].keys()),
            "metabolites_with_both_annotations": len(joint),
            "parent_bbb_negative_metabolite_positive_pairs": len(candidate_pairs),
            "families_with_parent_negative_metabolite_positive_pairs": len({families[e["parent_inchikey"]] for e in candidate_pairs}),
            "candidate_reversal_pairs_with_direct_metabolism_doi": sum(
                any(evidence.get("source_doi") for evidence in edge["evidence"]) for edge in candidate_pairs),
        },
        "chemistry_diagnostics": {
            "invalid_unique_supplied_structures": len(identities.invalid),
            "multi_fragment_unique_molecules": sum(row["fragment_count"] > 1 for row in identities.molecules.values()),
            "connectivity_groups_with_multiple_full_identities": sum(len(values) > 1 for values in connectivity.values()),
        },
        "blocking_findings": [
            "Metabolite identity matches are candidate benchmark annotations; underlying publication, assay, species and independence require review before cohort acceptance.",
            "Source names reference/test describe upstream metabolism partitions. Novelty 3 reserves family and scaffold groups during data preparation.",
            "Human metabolism is reported at source level; the downloaded relationship files do not provide complete per-reaction species and experimental citations.",
            "The P-gp CSV supplies binary substrate labels without per-record assay context. Quantitative efflux or clinical exposure cannot be inferred.",
        ],
        "limitations": [
            "BBB annotations are heterogeneous and do not isolate passive permeability.",
            "Source annotation matches are not newly acquired experimental validation or proof of human brain exposure.",
            "Unknown or conflicting measurements are excluded from class counts and remain explicitly unknown in coverage outputs.",
            "Strict identity matching retains salts and stereochemistry; possible reconciliations require review and are not automatically merged.",
            "Source datasets may reuse publications. Duplicate records cannot establish independent evidence.",
            "Documented metabolite absence from the sources does not establish absence of metabolism.",
            "Source usage notices are retained; redistribution terms require checking before publishing a dataset.",
            "Family components must also be joined by scaffold before the final evaluation split; independent component counts here are an upper bound on scaffold-independent groups.",
        ],
    }
    if not outcomes["bbb"]["labelled_metabolites"]:
        report["blocking_findings"].insert(0, "No BBB-labelled metabolites were matched; the proposed primary evaluation cannot yet be constructed.")
    if not outcomes["bbb"]["positive_metabolites"] or not outcomes["bbb"]["negative_metabolites"]:
        report["blocking_findings"].insert(0, "Both BBB classes are not represented among matched metabolites; class-based metabolite evaluation is incomplete.")
    if candidate_pairs and not report["annotation_intersections"]["candidate_reversal_pairs_with_direct_metabolism_doi"]:
        report["blocking_findings"].insert(0, "The candidate BBB-negative precursor / BBB-positive metabolite pairs lack direct metabolism publication DOIs in the downloaded relationships. Their underlying studies must be resolved before supporting the central hypothesis.")
    return report, pair_rows, families


def render_report(report):
    lines = ["# Novelty 3 phase 1 feasibility audit", "", "Status is REVIEW_REQUIRED. Model fitting has not been performed.", "",
             "## What the counts mean", "",
             "Counts describe strict molecular-identity matches to source annotations. They are candidates for manual evidence review. Parent–metabolite relationships do not supply transport labels.", "",
             "## Endpoint coverage", ""]
    for endpoint, values in report["endpoint_coverage"].items():
        lines.extend([
            f"### {endpoint.upper()}", "",
            f"- Eligible unique endpoint-labelled molecules — {values['usable_unique_molecules']}",
            f"- Conflicting molecular identities excluded — {values['conflicting_molecule_count']}",
            f"- Repeated identity/label rows collapsed for molecule counts — {values['repeated_identity_label_rows']}",
            f"- Matched labelled metabolites — {values['labelled_metabolites']}",
            f"- Positive / negative metabolites — {values['positive_metabolites']} / {values['negative_metabolites']}",
            f"- Family components with labelled metabolites — {values['independent_components_with_labelled_metabolites']}",
            f"- Manually curated experimental metabolite identities — {values['manually_curated_labelled_metabolites']}", "",
        ])
    lines.extend(["## Documented relationships", ""])
    lines.extend(f"- {name.replace('_', ' ')} — {value}" for name, value in report["metabolism"].items())
    lines.extend(["", "## Candidate overlap", ""])
    lines.extend(f"- {name.replace('_', ' ')} — {value}" for name, value in report["annotation_intersections"].items())
    lines.extend(["", "## Evidence review for scientific claims", ""])
    lines.extend(f"- {finding}" for finding in report["blocking_findings"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {finding}" for finding in report["limitations"])
    lines.extend(["", "## Next research step", "",
                  "Run 'data prepare' for the retrospective benchmark. Review source citations, assay definitions and the candidate reversal pairs before making experimental or human CNS exposure claims.", ""])
    return "\n".join(lines)


def perform_audit():
    RDLogger.DisableLog("rdApp.*")
    sources = verified_sources()
    identities = IdentityStore(ROOT / "data" / "cache" / "identities.json")
    measurements = []
    statistics = {}
    for endpoint in ("bbb", "pgp"):
        print(f"[audit] Reading {endpoint} measurements", flush=True)
        rows, counts = load_endpoint(endpoint, identities)
        measurements.extend(rows)
        statistics[endpoint] = counts
        identities.save()
    curation_path = ROOT / "data" / "input" / "metabolite_measurements.csv"
    print("[audit] Reading manually curated measurements", flush=True)
    curated = load_curated(curation_path, identities)
    measurements.extend(curated)
    statistics["manually_curated_records"] = len(curated)
    print("[audit] Resolving documented metabolic relationships", flush=True)
    edges, relationship_statistics = load_relationships(identities)
    statistics.update(relationship_statistics)
    identities.save()
    report, coverage, families = summarize(measurements, edges, identities, statistics)
    report["sources"] = sources
    report["curation_sha256"] = digest(curation_path.read_bytes())
    report["audit_version"] = "0.1.0"
    processed = ROOT / "data" / "processed"
    results = ROOT / "results" / "phase1"
    write_json(processed / "molecules.json", {"policy": IDENTITY_POLICY, "molecules": identities.molecules})
    write_json(processed / "relationships.json", edges)
    write_json(processed / "measurements.json", measurements)
    family_path = processed / "family_components.json"
    # Preserve an identical legacy mapping's bytes because its insertion order
    # was part of the dataset fingerprint used by checkpoints.
    try:
        previous_families = json.loads(family_path.read_bytes())
    except (FileNotFoundError, ValueError):
        previous_families = None
    if previous_families != families:
        write_json(family_path, families)
    write_json(processed / "invalid_structures.json", identities.invalid)
    fields = ("family_id", "parent_inchikey", "metabolite_inchikey", "parent_smiles", "metabolite_smiles", "parent_bbb_label",
              "metabolite_bbb_label", "parent_pgp_substrate_label", "metabolite_pgp_substrate_label",
              "source_ids", "metabolism_references", "evidence_status")
    write_csv(results / "candidate_coverage.csv", coverage, fields)
    atomic_bytes(results / "feasibility_report.md", render_report(report).encode("utf-8"))
    write_json(results / "feasibility_report.json", report)
    print("[complete] Audit status REVIEW_REQUIRED. Run 'data prepare' for the retrospective benchmark.")
    for task, values in report["endpoint_coverage"].items():
        print(f"  {task}: {values['usable_unique_molecules']} endpoint-labelled molecules; "
              f"{values['labelled_metabolites']} matched metabolites in "
              f"{values['independent_components_with_labelled_metabolites']} family components")
    print(f"  Report: {results / 'feasibility_report.md'}")
    return 2


def run_audit():
    """Publish a completion manifest last so partial or failed runs are explicit."""
    manifest_path = ROOT / "results" / "phase1" / "audit_run.json"
    started = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, {"status": "RUNNING", "started_at": started})
    try:
        exit_code = perform_audit()
        artifacts = (
            "data/processed/molecules.json", "data/processed/relationships.json",
            "data/processed/measurements.json", "data/processed/family_components.json",
            "data/processed/invalid_structures.json", "results/phase1/candidate_coverage.csv",
            "results/phase1/feasibility_report.md", "results/phase1/feasibility_report.json",
        )
        checksums = {name: digest((ROOT / name).read_bytes()) for name in artifacts}
        write_json(manifest_path, {
            "status": "REVIEW_REQUIRED", "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": checksums, "exit_code": exit_code,
        })
        return exit_code
    except (Exception, KeyboardInterrupt) as error:
        write_json(manifest_path, {
            "status": "INCOMPLETE", "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}",
            "action": "Rerun data audit. Previous report files do not represent a completed current run.",
        })
        raise
