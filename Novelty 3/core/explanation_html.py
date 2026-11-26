"""Render portable, self-contained HTML reports for explanation evidence."""

from html import escape
import json

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D

from .explanations import ROLES


ENDPOINT_NAMES = {"bbb": "BBB", "pgp": "P-gp"}
STAGE_NAMES = {"bbb_rejected": "BBB rejected", "pgp_flagged": "P-gp flagged", "pass": "Passed both gates"}
SKIP_NAMES = {"bbb_rejected_pgp_skipped": "P-gp skipped after BBB rejection",
              "fewer_than_three_bonds": "Fewer than three bonds"}
CSS = """
body { font: 16px/1.55 system-ui, sans-serif; margin: 0; background: #f3f5f7; color: #162333; }
main { max-width: 1100px; margin: auto; padding: 24px; }
h1, h2, h3 { line-height: 1.2; }
h2 { margin-top: 0; } h3 { margin: 0.7em 0; }
section { background: white; border: 1px solid #d3dbe1; border-radius: 10px; padding: 20px; margin: 16px 0; }
.pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.pair > article { border-top: 2px solid #d3dbe1; min-width: 0; }
.scroll, pre { overflow: auto; }
svg { display: block; height: auto; max-width: 100%; margin: 0 auto; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; font-variant-numeric: tabular-nums; }
td, th { padding: 6px 10px; text-align: left; border-bottom: 1px solid #d3dbe1; }
th { background: #edf2f5; } code { overflow-wrap: anywhere; }
small { color: #465565; } summary { cursor: pointer; font-weight: 600; }
.positive { color: #1358a6; } .negative { color: #b13427; } .unmatched { color: #8a580a; }
@media (max-width: 720px) { .pair { grid-template-columns: 1fr; } main, section { padding: 12px; } }
"""


def page(title, content):
    return (
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width, initial-scale=1"><title>{escape(title)}</title>'
        f'<style>{CSS}</style><main><h1>{escape(title)}</h1>{content}</main></html>'
    )


def table(headings, rows):
    def cells(tag, values):
        return "".join(f"<{tag}>{escape(str(value))}</{tag}>" for value in values)

    body = "".join(f"<tr>{cells('td', row)}</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{cells("th", headings)}</tr></thead><tbody>{body}</tbody></table></div>'


def evidence_details(report):
    details = escape(json.dumps(report["provenance"], indent=2))
    return f"<details><summary>Dataset and checkpoint identifiers</summary><pre>{details}</pre></details>"


def molecule_svg(mol, atom_colors=None, bond_colors=None, labels=None, bounds=None):
    drawing = Chem.Mol(mol)
    for index, note in (labels or {}).items():
        drawing.GetBondWithIdx(index).SetProp("bondNote", note)
    drawer = rdMolDraw2D.MolDraw2DSVG(460, 290)
    drawer.drawOptions().clearBackground = False
    if bounds:
        drawer.SetScale(460, 290, *bounds)
    drawer.DrawMolecule(drawing, highlightAtoms=list(atom_colors or {}), highlightBonds=list(bond_colors or {}),
                        highlightAtomColors=atom_colors or {}, highlightBondColors=bond_colors or {})
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index("<svg"):]


def columns(molecules, render):
    items = (f"<article><h3>{role.title()}</h3>{render(role, mol)}</article>" for role, mol in zip(ROLES, molecules))
    return '<div class="pair">' + "".join(items) + "</div>"


def structure_html(comparison, molecules):
    if comparison["status"] == "unavailable":
        note = f"<p>Structural comparison unavailable. {escape(comparison['reason'])}</p>"
        return note + columns(molecules, lambda role, mol: molecule_svg(mol))
    note = '<p><strong class="unmatched">Amber marks unmatched atoms and bonds.</strong> Shared structure coverage is measured over heavy atoms.</p>'
    positions = [point for mol in molecules for point in mol.GetConformer().GetPositions()]
    # Leave room for labels and keep matched atoms at the same position in both panels.
    xs, ys = zip(*(point[:2] for point in positions))
    bounds = (Point2D(min(xs) - 1, min(ys) - 1), Point2D(max(xs) + 1, max(ys) + 1))

    def depiction(role, mol):
        unmatched = comparison["unmatched"][role]
        amber = (0.97, 0.72, 0.27)
        coverage = comparison["heavy_atom_coverage"][role]
        return (
            molecule_svg(mol, dict.fromkeys(unmatched["atom_indices"], amber),
                         dict.fromkeys(unmatched["bond_indices"], amber), bounds=bounds)
            + f"<p>{coverage:.0%} heavy-atom coverage</p>"
            + f"<p>Unmatched atom indices {unmatched['atom_indices']} · bond indices {unmatched['bond_indices']}</p>"
        )

    return note + columns(molecules, depiction)


def gate_html(mol, entry, scale):
    if entry["status"] == "skipped":
        return f'<p><strong>{escape(entry["reason"])}</strong> · cutoff {entry["cutoff"]:.6f}</p>'
    decision = {"pass": "Gate passed", **{k: v for k, v in STAGE_NAMES.items() if k != "pass"}}[entry["decision"]]
    intro = (
        f'<p><strong>{decision}</strong> · probability {entry["probability"]:.6f} · cutoff {entry["cutoff"]:.6f}'
        f' · margin {entry["margin"]:+.6f} · seed SD {entry["seed_sd"]:.6f}</p>'
    )
    seed_probabilities = " · ".join(f"{seed}: {p:.6f}" for seed, p in entry["seed_probabilities"].items())
    intro += f"<p><small>Seed probabilities · {seed_probabilities}</small></p>"
    if entry["bond_effects_unavailable"]:
        return intro + f'<p>{escape(entry["bond_effects_unavailable"])}</p>' + molecule_svg(mol)

    colors = {}
    for row in entry["bond_effects"]:
        value = row["mean_effect"]
        if value:
            base = (0.08, 0.36, 0.72) if value > 0 else (0.78, 0.18, 0.14)
            strength = abs(value) / scale
            colors[row["bond_index"]] = tuple(1 - strength * (1 - component) for component in base)
    labels = {row["bond_index"]: str(row["bond_index"]) for row in entry["bond_effects"][:5]}
    rows = [
        (row["bond_index"], "–".join(f"{el}{i}" for el, i in zip(row["elements"], row["atom_indices"])),
         row["bond_type"], f'{row["mean_effect"]:+.6f}', f'{row["directional_agreement"]:.0%}',
         *(f"{value:+.6f}" for value in row["seed_effects"].values()))
        for row in entry["bond_effects"]
    ]
    return intro + molecule_svg(mol, bond_colors=colors, labels=labels) + table(
        ("Bond", "Atoms", "Type", "Mean Δ", "Agreement", *entry["seed_probabilities"]), rows
    )


def pair_html(report, molecules):
    entries = report["molecules"]
    relative = report["ranking"]["parent_relative_support"]
    ranking = ("The metabolite is unranked after rejection." if relative is None
               else f"Parent-relative support is <strong>{relative:+.6f}</strong> (metabolite support minus parent support).")
    if report["ranking"]["parent_zero_after_rejection"]:
        ranking += " The rejected parent contributes zero support, so the entire relative score comes from the metabolite."

    def decision(role, mol):
        entry = entries[role]
        support = entry["support"]
        detail = "Zero after rejection."
        if "limiting_term" in support:
            limiting = {"bbb": "BBB", "pgp": "P-gp", "both": "both gates", "floor": "zero floor"}[support["limiting_term"]]
            detail = (f'Adjusted BBB {support["bbb_adjusted_support"]:.6f} · adjusted P-gp nonsubstrate '
                      f'{support["pgp_adjusted_support"]:.6f} · limiting term {limiting}.')
        return (
            f'<code>{escape(entry["inchikey"])}</code><p><strong>{STAGE_NAMES[entry["stage"]]}</strong>'
            f' · ranking support {support["value"]:.6f}. {detail}</p>'
        )

    endpoint_sections = []
    for endpoint, title in ENDPOINT_NAMES.items():
        scale = max((abs(bond["mean_effect"]) for entry in entries.values()
                     for bond in entry["endpoints"][endpoint].get("bond_effects", [])), default=0)
        panels = columns(molecules, lambda role, mol: gate_html(mol, entries[role]["endpoints"][endpoint], scale))
        event = "BBB positive" if endpoint == "bbb" else "P-gp substrate"
        endpoint_sections.append(f"<section><h2>{title} bond sensitivity</h2><p>Probability of {event}</p>{panels}</section>")
    content = (
        f"<section><h2>Screening and ranking</h2>{columns(molecules, decision)}<p>{ranking}</p>"
        "<p>For a two-gate pass, support is the smaller of BBB-positive probability and P-gp nonsubstrate "
        "probability after subtracting the corresponding seed SD, with a floor of zero.</p>"
        f'<p>Family <code>{escape(report["family_id"])}</code> · merged group <code>{escape(report["group_id"])}</code>'
        f' · reserved {"yes" if report["reserved_group"] else "no"}</p></section>'
        f'<section><h2>Shared structure</h2>{structure_html(report["structural_comparison"], molecules)}</section>'
        '<section><h2>Reading the bond effects</h2><p>Δ is the original calibrated seed probability minus the masked '
        'probability. <strong class="positive">Blue</strong> marks positive mean changes and '
        '<strong class="negative">red</strong> marks negative mean changes. Color intensity uses the same scale '
        'for both molecules within each endpoint. The five largest absolute effects are labeled with zero-based '
        'bond indices. Agreement is the fraction of seeds matching the sign of the mean, including exact zeros.</p>'
        '<p>Masking removes both message edges for a bond while preserving the original atom features. This graph '
        'diagnostic can expose the model to graphs that differ from its training data. Establishing a biological '
        'mechanism or improved brain exposure requires matched experimental evidence.</p></section>'
        + "".join(endpoint_sections) + evidence_details(report)
    )
    return page("Parent–metabolite explanation", content)


def benchmark_html(report):
    design, pool = report["design"], report["pool"]
    counts = table(("Metabolite outcome", "Pool pairs", "Pool groups", "Selected", "Shortfall"),
                   [(STAGE_NAMES[stage], *[row[f] for f in ("candidate_pairs", "candidate_groups", "selected", "shortfall")])
                    for stage, row in report["cohort_counts"].items()])
    metrics = []
    for endpoint, row in report["endpoints"].items():
        interval = row["paired_group_bootstrap_95_ci"]
        skips = ", ".join(f"{SKIP_NAMES[key]} ({count})" for key, count in row["skipped_by_reason"].items())
        fields = ("top_abs_change", "random_mean_abs_change", "effect_over_random",
                  "seed_top_bond_overlap", "directional_agreement")
        top, random, delta, overlap, agreement = [
            "Unavailable" if row[field] is None else format(row[field], spec)
            for field, spec in zip(fields, (".6f", ".6f", "+.6f", ".1%", ".1%"))
        ]
        metrics.append((ENDPOINT_NAMES[endpoint], row["eligible_groups"], row["eligible_cases"], row["skipped_cases"],
                        skips, top, random, delta, "Descriptive only" if interval is None else
                        f"[{interval[0]:+.6f}, {interval[1]:+.6f}]", overlap, agreement))
    results = table(("Endpoint", "Eligible groups", "Eligible cases", "Skipped cases", "Skip reasons",
                     "Top |Δ|", "Random |Δ|", "Effect over random", "Paired 95% interval",
                     "Top-bond overlap", "Directional agreement"), metrics)
    cases = table(("Outcome", "Merged group", "Parent", "Metabolite"),
                  [(STAGE_NAMES[row["metabolite_stage"]], row["group_id"], row["parent"], row["metabolite"])
                   for row in report["cohort"]])
    content = (
        f"<section><h2>Held-out cohort</h2><p>{pool['pairs']} documented pool pairs in {pool['groups']} "
        f"reserved merged groups after excluding model development groups. The sample includes up to "
        f"{design['max_per_stage']} pairs per outcome and one per merged group. Pair IDs are hashed with "
        f"seed {design['seed']} and selected in the order shown.</p>{counts}</section>"
        f"<section><h2>Bond masking results</h2>{results}<p>Four seeds rank single-bond effects by absolute mean. "
        "The fifth seed tests the selected bonds jointly. All five seeds take a turn as the held-out seed. "
        "The selected mask covers 20% of bonds rounded up, with at least two masked and one retained.</p>"
        f"<p>Each rotation compares the absolute probability change with the mean absolute change across "
        f"{design['random_masks_per_rotation']} uniform masks of the same size. The reported difference averages "
        f"rotations and eligible molecules within each group, weighting groups equally. The paired interval uses "
        f"{design['bootstrap_resamples']:,} bootstrap resamples of groups and requires at least two groups.</p>"
        "<p>Top-bond overlap is the mean fraction shared between individual seeds’ top lists. Directional agreement "
        "compares the signs of the four-seed mean and held-out single-bond effects on the selected bonds.</p></section>"
        f"<section><h2>Selected pairs</h2>{cases}</section>" + evidence_details(report)
    )
    return page("Held-out bond sensitivity benchmark", content)
