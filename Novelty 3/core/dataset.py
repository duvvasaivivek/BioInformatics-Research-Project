"""Prepare one versioned graph cache and independent endpoint partitions."""

from collections import Counter
import io
import json

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch_geometric.data import Data

from .audit import endpoint_labels
from .sources import digest
from .storage import ROOT, atomic_bytes

DATA_VERSION = 1
ENDPOINTS = ("bbb", "pgp")
SPLITS = ("train", "validation", "calibration", "test")
ATOM_DIM, BOND_DIM = 12, 7
AUDIT_MANIFEST = ROOT / "results" / "phase1" / "audit_run.json"
CURATION = ROOT / "data" / "input" / "metabolite_measurements.csv"
CACHE = ROOT / "data" / "cache" / "training.pt"


def save_torch(path, value):
    stream = io.BytesIO()
    torch.save(value, stream)
    atomic_bytes(path, stream.getvalue())


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def audit_inputs():
    manifest = json.loads(AUDIT_MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "REVIEW_REQUIRED":
        raise ValueError("Complete 'data audit' before preparing the datasets.")
    names = ("molecules", "measurements", "relationships", "family_components")
    paths = {name: f"data/processed/{name}.json" for name in names}
    paths["report"] = "results/phase1/feasibility_report.json"
    contents = {}
    checksums = {}
    for name, path in paths.items():
        raw = (ROOT / path).read_bytes()
        checksum = digest(raw)
        if checksum != manifest["artifact_sha256"].get(path):
            raise ValueError(f"Audit artifact changed: {path}. Rerun 'data audit'.")
        contents[name] = json.loads(raw)
        if name != "report":
            checksums[name] = checksum
    curation_sha = digest(CURATION.read_bytes())
    if contents["report"]["curation_sha256"] != curation_sha:
        raise ValueError("Experimental curation changed. Rerun 'data audit'.")
    stamp = {"files": checksums, "curation": curation_sha, "rdkit": rdBase.rdkitVersion,
             "version": DATA_VERSION}
    return contents, digest(json.dumps(stamp, sort_keys=True).encode())


def molecule_graph(mol):
    atoms = []
    for atom in mol.GetAtoms():
        chirality = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        atoms.append([
            atom.GetAtomicNum() / 100, atom.GetTotalDegree() / 6,
            atom.GetFormalCharge() / 4, atom.GetTotalNumHs() / 4,
            float(atom.GetIsAromatic()), float(atom.IsInRing()),
            float(atom.GetHybridization() == Chem.HybridizationType.SP),
            float(atom.GetHybridization() == Chem.HybridizationType.SP2),
            float(atom.GetHybridization() == Chem.HybridizationType.SP3),
            float(chirality == "R"), float(chirality == "S"), atom.GetMass() / 250,
        ])
    edges, features = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row = [bond.GetBondTypeAsDouble() / 3, float(bond.GetIsAromatic()),
               float(bond.GetIsConjugated()), float(bond.IsInRing()),
               float(bond.GetStereo() == Chem.BondStereo.STEREOE),
               float(bond.GetStereo() == Chem.BondStereo.STEREOZ),
               float(bond.GetStereo() == Chem.BondStereo.STEREOANY)]
        edges.extend(((start, end), (end, start)))
        features.extend((row, row))
    return (torch.tensor(atoms, dtype=torch.float32).reshape(-1, ATOM_DIM),
            torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
            torch.tensor(features, dtype=torch.float32).reshape(-1, BOND_DIM))


def graph_data(cache, key, endpoint):
    x, edge_index, edge_attr = cache["graphs"][key]
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([cache["labels"][endpoint].get(key, -1)], dtype=torch.float32))


def leakage_groups(molecules, families, parsed):
    """Join by family, nonempty scaffold, and fragment connectivity."""
    root = {key: key for key in molecules}
    seen = {}

    def find(key):
        while key != root[key]:
            root[key] = root[root[key]]
            key = root[key]
        return key

    for key in sorted(molecules):
        # Strip counterions for grouping only. Graphs and labels keep the audited identity.
        fragments = Chem.GetMolFrags(parsed[key], asMols=True)
        main_size = max(fragment.GetNumHeavyAtoms() for fragment in fragments)
        tokens = [("connectivity", molecules[key]["connectivity_key"])]
        if key in families:
            tokens.append(("family", families[key]))
        for fragment in fragments:
            if fragment.GetNumHeavyAtoms() != main_size:
                continue
            Chem.RemoveStereochemistry(fragment)
            tokens.append(("fragment", Chem.MolToSmiles(fragment, isomericSmiles=False)))
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=fragment)
            if scaffold:
                tokens.append(("scaffold", scaffold))
        for token in tokens:
            other = find(seen.setdefault(token, key))
            current = find(key)
            if other != current:
                # Stable IDs, including when a molecule bridges two existing groups.
                root[max(other, current)] = min(other, current)
    return {key: find(key) for key in molecules}


def endpoint_split(labels, groups, reserved_groups):
    keys = [key for key in sorted(labels) if groups[key] not in reserved_groups]
    target = np.asarray([labels[key] for key in keys])
    group_ids = np.asarray([groups[key] for key in keys])
    if len(set(group_ids)) < 10 or min(Counter(target).get(c, 0) for c in (0, 1)) < 10:
        raise ValueError("Too few eligible groups or classes for a ten-fold grouped partition.")
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=2026)
    folds = [part for _, part in splitter.split(target, target, group_ids)]
    chosen = [part for part in folds if len(set(target[part])) == 2][:3]
    if len(chosen) != 3:
        raise ValueError("Could not create three grouped holdouts containing both classes.")
    train = sorted(set(range(len(keys))) - {int(i) for part in chosen for i in part})
    splits = {name: [keys[int(i)] for i in part] for name, part in zip(SPLITS, [train, *chosen])}
    for name, members in splits.items():
        if {labels[key] for key in members} != {0, 1}:
            raise ValueError(f"Partition {name} requires both labels.")
    return splits


def partition_summary(cache):
    summary = {}
    for endpoint, parts in cache["splits"].items():
        labels = cache["labels"][endpoint]
        reserved = sorted(key for key in labels if cache["groups"][key] in cache["reserved_groups"])
        summary[endpoint] = {
            name: {"molecules": len(keys), "positive": sum(labels[key] for key in keys),
                   "groups": len({cache["groups"][key] for key in keys})}
            for name, keys in {**parts, "reserved": reserved}.items()
        }
    return summary


def prepare():
    inputs, data_id = audit_inputs()
    if CACHE.exists():
        cached = load_torch(CACHE)
        if cached["data_id"] == data_id:
            print("Reusing graph cache and endpoint partitions.")
            print(json.dumps(partition_summary(cached), indent=2))
            return cached
    labels, _ = endpoint_labels(inputs["measurements"])
    families = inputs["family_components"]
    # Source diagnostics may include molecules with no label or relationship.
    keys = set(families) | labels["bbb"].keys() | labels["pgp"].keys()
    molecules = {key: inputs["molecules"]["molecules"][key] for key in sorted(keys)}
    parsed = {key: Chem.MolFromSmiles(row["canonical_smiles"]) for key, row in molecules.items()}
    if any(mol is None or not mol.GetNumAtoms() for mol in parsed.values()):
        raise ValueError("An audited canonical structure could not be parsed by the current RDKit.")
    groups = leakage_groups(molecules, families, parsed)
    metabolites = {edge["metabolite_inchikey"] for edge in inputs["relationships"]}
    reserved = sorted({groups[key] for key in metabolites & (labels["bbb"].keys() | labels["pgp"].keys())})
    splits = {task: endpoint_split(labels[task], groups, set(reserved)) for task in ENDPOINTS}
    curated_rows = [row for row in inputs["measurements"]
                    if row["evidence_level"] == "manually_curated_experiment"]
    curated = {task: sorted({row["inchikey"] for row in curated_rows
                            if row["endpoint"] == task and row["inchikey"] in labels[task]})
               for task in ENDPOINTS}
    cache = {"data_id": data_id, "splits": splits, "labels": labels,
             "groups": groups, "reserved_groups": reserved, "families": families,
             "curated": curated, "relationships": [
                 [row["parent_inchikey"], row["metabolite_inchikey"]] for row in inputs["relationships"]
             ], "graphs": {key: molecule_graph(mol) for key, mol in parsed.items()}}
    save_torch(CACHE, cache)
    print(f"Prepared {len(parsed)} graphs and {len(reserved)} reserved evaluation groups.")
    print(json.dumps(partition_summary(cache), indent=2))
    return cache


def load_dataset():
    if not CACHE.exists():
        raise ValueError("Run 'data prepare' first.")
    _, data_id = audit_inputs()
    cache = load_torch(CACHE)
    if cache["data_id"] != data_id:
        raise ValueError("The graph cache is stale. Rerun 'data prepare'.")
    return cache
