"""Trace frozen screening decisions and measure sensitivity to bond masking."""

from copy import copy
from hashlib import sha256
import json
from time import monotonic

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdFMCS
import torch
from torch_geometric.loader import DataLoader

from .dataset import ENDPOINTS, graph_data, load_dataset, molecule_graph
from .learning import SEEDS, calibrated_ensemble, calibrated_probability, device_for, predict_logits, probabilities
from .screening import relationship_rows, sequential_scores
from .storage import ROOT, atomic_bytes, write_json


RESULTS = ROOT / "results" / "explanations"
STAGES = ("bbb_rejected", "pgp_flagged", "pass")
PROBABILITY_FIELDS = {"bbb": "bbb_probability", "pgp": "pgp_substrate_probability"}
ROLES = ("parent", "metabolite")


def documented_pair(cache, parent, metabolite):
    pair = (parent, metabolite)
    if not any(tuple(relationship) == pair for relationship in cache["relationships"]):
        raise ValueError("Supply the exact parent and metabolite InChIKeys of a documented pair, in that order.")
    return pair


def verified_ensembles(cache, device):
    ensembles = {endpoint: calibrated_ensemble(cache, endpoint, device) for endpoint in ENDPOINTS}
    for endpoint, (_, calibration) in ensembles.items():
        if tuple(sorted(calibration["seeds"])) != SEEDS:
            raise ValueError(f"The {endpoint} explanation requires the five frozen seeds {SEEDS}.")
    return ensembles


def provenance(cache, ensembles, device):
    return {
        "data_id": cache["data_id"], "device": str(device),
        "endpoints": {
            endpoint: {
                "seeds": report["seeds"], "model_version": report["model_version"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "calibration_sha256": sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
                "slope": report["slope"], "intercept": report["intercept"], "threshold": report["threshold"],
            }
            for endpoint, (_, report) in ensembles.items()
        },
    }


def checked_structure(cache, key):
    """Recover the graph's atom order and verify the exact edge/feature correspondence."""
    mol = Chem.MolFromSmiles(cache["structures"][key])
    if mol is None or not all(torch.equal(expected, cached)
                              for expected, cached in zip(molecule_graph(mol), cache["graphs"][key])):
        raise ValueError(f"Drawing-to-graph atom and bond mapping could not be verified for {key}.")
    bonds = []
    for bond in mol.GetBonds():
        index = bond.GetIdx()
        bonds.append({
            "bond_index": index, "atom_indices": [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
            "elements": [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()],
            "bond_type": str(bond.GetBondType()), "directed_edge_indices": [2 * index, 2 * index + 1],
        })
    return mol, bonds


def masked_graph(graph, bonds, mask):
    keep = torch.ones(graph.edge_index.shape[1], dtype=torch.bool)
    for index in mask:
        keep[bonds[index]["directed_edge_indices"]] = False
    masked = copy(graph)
    masked.edge_index = graph.edge_index[:, keep]
    masked.edge_attr = graph.edge_attr[keep]
    return masked


def replay(model, graph, bonds, masks, device):
    variants = [masked_graph(graph, bonds, mask) if mask else graph for mask in masks]
    loader = DataLoader(variants, batch_size=64, num_workers=0, generator=torch.Generator().manual_seed(0))
    return predict_logits(model, loader, device).astype(np.float64)


def gate_result(endpoint, probability, cutoff):
    if endpoint == "bbb":
        return "pass" if probability >= cutoff else "bbb_rejected"
    return "pgp_flagged" if probability >= cutoff else "pass"


def check_replay(endpoint, original_logits, calibration, screening):
    centre, spread = probabilities(original_logits, calibration)
    expected = screening[PROBABILITY_FIELDS[endpoint]]
    cutoff = calibration["threshold"]
    decision = "bbb_rejected" if endpoint == "bbb" and screening["stage"] == "bbb_rejected" else (
        "pass" if endpoint == "bbb" else screening["stage"])
    if (not np.allclose([centre, spread], [expected, screening[f"{endpoint}_seed_sd"]], rtol=1e-5, atol=1e-7)
            or gate_result(endpoint, centre, cutoff) != decision):
        raise ValueError(f"Unmasked {endpoint} replay did not reproduce the ordinary screening score and gate.")


def single_bond_replay(cache, key, endpoint, ensemble, device, bonds, screening):
    models, calibration = ensemble
    graph = graph_data(cache, key, endpoint)
    masks = [(), *((bond["bond_index"],) for bond in bonds)]
    logits = np.stack([replay(model, graph, bonds, masks, device) for model in models])
    check_replay(endpoint, logits[:, 0], calibration, screening)
    per_seed = calibrated_probability(logits, calibration)
    return graph, per_seed[:, 0], per_seed[:, :1] - per_seed[:, 1:]


def seed_agreement(effects):
    """Fraction of all seeds with a sign equal to the mean sign, including ties at zero."""
    direction = np.sign(np.mean(effects, axis=0))
    return np.mean(np.sign(effects) == direction, axis=0)


def explain_endpoint(cache, key, endpoint, ensemble, device, bonds, screening):
    cutoff = ensemble[1]["threshold"]
    probability = screening[PROBABILITY_FIELDS[endpoint]]
    if probability is None:
        return {"status": "skipped", "reason": "P-gp skipped (BBB rejected)", "cutoff": cutoff}
    _, original, effects = single_bond_replay(cache, key, endpoint, ensemble, device, bonds, screening)
    seeds = ensemble[1]["seeds"]
    rows = [
        {**bond, "mean_effect": float(mean), "directional_agreement": float(agreement),
         "seed_effects": dict(zip(map(str, seeds), map(float, changes)))}
        for bond, mean, agreement, changes in zip(bonds, effects.mean(axis=0), seed_agreement(effects), effects.T)
    ]
    rows.sort(key=lambda row: (-abs(row["mean_effect"]), row["bond_index"]))
    return {
        "status": "scored", "decision": gate_result(endpoint, probability, cutoff),
        "probability": probability, "cutoff": cutoff, "margin": probability - cutoff,
        "seed_sd": screening[f"{endpoint}_seed_sd"],
        "seed_probabilities": dict(zip(map(str, seeds), map(float, original))),
        "bond_effects": rows, "bond_effects_unavailable": None if bonds else "No chemical bonds to mask.",
    }


def structural_alignment(parent, metabolite):
    parameters = rdFMCS.MCSParameters()
    parameters.Timeout = 5
    parameters.StoreAll = True
    parameters.AtomTyper = rdFMCS.AtomCompare.CompareElements
    parameters.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    parameters.AtomCompareParameters.MatchChiralTag = True
    parameters.BondCompareParameters.MatchStereo = True
    for compare in (parameters.AtomCompareParameters, parameters.BondCompareParameters):
        compare.RingMatchesRingOnly = True
        compare.CompleteRingsOnly = True
    molecules = (parent, metabolite)
    rdDepictor.Compute2DCoords(parent)
    rdDepictor.Compute2DCoords(metabolite)
    deadline = monotonic() + parameters.Timeout
    result = rdFMCS.FindMCS(molecules, parameters)
    if result.canceled:
        return {"status": "unavailable", "reason": "Shared structure search reached the five-second timeout."}
    queries = result.degenerateSmartsQueryMolDict
    # RDKit can leave StoreAll empty for a single-atom result.
    if result.numAtoms == 1 and not queries:
        parameters.StoreAll = False
        remaining = int(deadline - monotonic())
        if remaining < 1:
            return {"status": "unavailable", "reason": "Shared structure search reached the five-second timeout."}
        parameters.Timeout = remaining
        result = rdFMCS.FindMCS(molecules, parameters)
        if result.canceled:
            return {"status": "unavailable", "reason": "Shared structure search reached the five-second timeout."}
        queries = {result.smartsString: result.queryMol}
    if not queries:
        return {"status": "unavailable", "reason": "No shared structure could be matched."}
    if len(queries) > 1:
        return {"status": "unavailable", "reason": "Multiple maximum shared structures make the alignment ambiguous."}
    query = next(iter(queries.values()))
    matches = [mol.GetSubstructMatches(query, useChirality=True, uniquify=False, maxMatches=2)
               for mol in molecules]
    if any(not match for match in matches):
        return {"status": "unavailable", "reason": "The shared structure could not be mapped with chirality."}
    if any(len(match) > 1 for match in matches):
        return {"status": "unavailable", "reason": "Multiple atom mappings make the alignment ambiguous."}
    indices = [match[0] for match in matches]
    fractions = [sum(mol.GetAtomWithIdx(i).GetAtomicNum() > 1 for i in mapped) / (mol.GetNumHeavyAtoms() or 1)
                 for mol, mapped in zip(molecules, indices)]
    coverage = dict(zip(ROLES, fractions))
    if min(fractions) < 0.5:
        return {"status": "unavailable", "reason": "Shared structure covers less than half of either molecule.",
                "heavy_atom_coverage": coverage}
    mapping = list(zip(*indices))
    try:
        rdDepictor.GenerateDepictionMatching2DStructure(metabolite, parent, mapping)
    except (ValueError, RuntimeError):
        rdDepictor.Compute2DCoords(metabolite)
        return {"status": "unavailable", "reason": "RDKit could not depict the verified atom alignment."}
    unmatched = {}
    for role, mol, mapped in zip(ROLES, molecules, indices):
        matched_bonds = {
            mol.GetBondBetweenAtoms(mapped[bond.GetBeginAtomIdx()], mapped[bond.GetEndAtomIdx()]).GetIdx()
            for bond in query.GetBonds()
        }
        unmatched[role] = {"atom_indices": sorted(set(range(mol.GetNumAtoms())) - set(mapped)),
                           "bond_indices": sorted(set(range(mol.GetNumBonds())) - matched_bonds)}
    return {"status": "available", "heavy_atom_coverage": coverage, "atom_mapping": mapping, "unmatched": unmatched}


def support_evidence(screening):
    if screening["stage"] != "pass":
        return {"value": screening["support"], "reason": "zero_after_rejection"}
    bbb = screening["bbb_probability"] - screening["bbb_seed_sd"]
    pgp = 1 - screening["pgp_substrate_probability"] - screening["pgp_seed_sd"]
    limiting = "floor" if min(bbb, pgp) < 0 else "bbb" if bbb < pgp else "pgp" if pgp < bbb else "both"
    return {"value": screening["support"], "bbb_adjusted_support": bbb, "pgp_adjusted_support": pgp,
            "limiting_term": limiting}


def explain_pair(parent, metabolite, choice):
    from .explanation_html import pair_html

    cache = load_dataset(include_structures=True)
    pair = documented_pair(cache, parent, metabolite)
    device = device_for(choice)
    ensembles = verified_ensembles(cache, device)
    scores = sequential_scores(cache, sorted(set(pair)), ensembles, device)
    row, = relationship_rows({**cache, "relationships": [pair]}, scores)
    molecules, structures = {}, []
    for role, key in zip(ROLES, pair):
        mol, bonds = checked_structure(cache, key)
        structures.append(mol)
        molecules[role] = {
            "inchikey": key, "canonical_smiles": cache["structures"][key],
            "stage": scores[key]["stage"], "support": support_evidence(scores[key]),
            "endpoints": {endpoint: explain_endpoint(cache, key, endpoint, ensembles[endpoint], device,
                                                     bonds, scores[key]) for endpoint in ENDPOINTS},
        }
    report = {
        "provenance": provenance(cache, ensembles, device), "family_id": row["family_id"],
        "group_id": row["group_id"], "reserved_group": row["reserved_group"], "molecules": molecules,
        "ranking": {
            "parent_relative_support": row["parent_relative_support"],
            "parent_zero_after_rejection": (row["parent_relative_support"] is not None
                                            and scores[parent]["stage"] != "pass"),
        },
        "structural_comparison": structural_alignment(*structures),
    }
    folder = RESULTS / f"{parent}__{metabolite}"
    html = pair_html(report, structures)
    write_json(folder / "explanation.json", report)
    atomic_bytes(folder / "report.html", html.encode("utf-8"))
    print(folder / "report.html")
    return report
