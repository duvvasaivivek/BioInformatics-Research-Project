"""Export figures from one saved Novelty 3 run without refitting or rescoring."""

import argparse
from collections import Counter
import csv
from io import BytesIO
import json
from math import isclose
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


ROOT = Path(__file__).resolve().parents[1]
WORKED_PAIR = "LMEKQMALGUDUQG-UHFFFAOYSA-N__GLVAUDGFNGKCSF-UHFFFAOYSA-N"
ENDPOINTS = ("bbb", "pgp")
ENDPOINT_LABELS = ("BBB", "P-gp")
STAGES = ("bbb_rejected", "pgp_flagged", "pass")
STAGE_LABELS = ("BBB rejected", "P-gp flagged", "Passed both")
INK, MUTED, GRID = "#172B40", "#465A70", "#DDE5ED"
BLUE, TEAL, AMBER, RED, GRAY = "#2667A6", "#087F79", "#A36B12", "#B43F46", "#CCD5E1"
STAGE_COLORS = dict(zip(STAGES, (RED, AMBER, TEAL)))
POLICY_COLORS = (TEAL, BLUE, AMBER)
POLICIES = ("parent_relative", "independent_metabolite", "parent_guided")
POLICY_LABELS = ("Parent-relative", "Metabolite-only", "Parent-guided")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": GRID,
    "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "pdf.fonttype": 42,
})


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_reports(results):
    evaluation = read_json(results / "evaluation.json")
    comparison = read_json(results / "pgp_comparison.json")
    pilot = read_json(results / "paired_validation.json")
    benchmark = read_json(results / "explanations" / "benchmark.json")
    pair = read_json(results / "explanations" / WORKED_PAIR / "explanation.json")
    data_id = evaluation["data_id"]
    if {data_id, comparison["data_id"], pilot["data_id"], benchmark["provenance"]["data_id"],
            pair["provenance"]["data_id"]} != {data_id}:
        raise ValueError("The saved reports come from different datasets. Supply a consistent results folder.")
    for endpoint in ENDPOINTS:
        saved = evaluation["calibration"][endpoint]["checkpoint_sha256"]
        candidates = (comparison[f"{endpoint}_gnn_calibration"], pilot["calibration"][endpoint],
                      benchmark["provenance"]["endpoints"][endpoint], pair["provenance"]["endpoints"][endpoint])
        fields = ("threshold", "slope", "intercept")
        calibration = evaluation["calibration"][endpoint]
        if any(candidate["checkpoint_sha256"] != saved
               or any(candidate[field] != calibration[field] for field in fields)
               for candidate in candidates):
            raise ValueError(f"The saved {endpoint} results use different checkpoint ensembles.")
    with (results / "screening.csv").open(encoding="utf-8", newline="") as stream:
        screening = list(csv.DictReader(stream))
    parent, metabolite = WORKED_PAIR.split("__")
    row, = (row for row in screening if (row["parent_inchikey"], row["metabolite_inchikey"]) == (parent, metabolite))
    for role, molecule in pair["molecules"].items():
        for endpoint, field in (("bbb", "bbb_probability"), ("pgp", "pgp_substrate_probability")):
            scored = molecule["endpoints"][endpoint].get("probability")
            recorded = row[f"{role}_{field}"]
            matches = not recorded if scored is None else recorded and isclose(scored, float(recorded), abs_tol=1e-6)
            if not matches or row[f"{role}_stage"] != molecule["stage"]:
                raise ValueError("The worked pair differs between screening.csv and explanation.json.")
    return data_id, screening, evaluation, comparison, pilot, benchmark, pair


def figure(title, subtitle, source, *, columns=2, rows=1, size=(10.8, 5.8)):
    fig, axes = plt.subplots(rows, columns, figsize=size, squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.22, h_pad=0.16)
    fig.suptitle(title, x=0.016, ha="left", fontsize=19, fontweight="bold")
    header = fig.supxlabel(subtitle + "\n\n" + source, fontsize=9.5, color=MUTED, x=0.016, ha="left")
    header.set_multialignment("left")
    return fig, axes.ravel()


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.7)


def save(fig, output, stem, data_id):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=240, facecolor="white",
                    metadata={"Title": fig.get_suptitle(), "Subject" if suffix == "pdf" else "Dataset": data_id})
    plt.close(fig)


def screening_figure(rows, data_id):
    fig, (ax,) = figure("Where the screen stops",
                        f"Each column covers the same {len(rows):,} documented relationships.",
                        f"Source  screening.csv  •  Rows are relationships, so a molecule may appear more than once."
                        f"  •  Dataset {data_id[:8]}",
                        columns=1, size=(8.8, 5.4))
    x = np.arange(2)
    bottom = np.zeros(2, dtype=int)
    counts = [Counter(row[f"{role}_stage"] for row in rows) for role in ("parent", "metabolite")]
    for stage, label in zip(STAGES, STAGE_LABELS):
        values = np.array([count[stage] for count in counts])
        bars = ax.bar(x, values, bottom=bottom, width=0.56, color=STAGE_COLORS[stage], label=label)
        ax.bar_label(bars, labels=[f"{v:,}  ({v / len(rows):.1%})" if v else "" for v in values],
                     label_type="center", color="white", fontweight="bold")
        bottom += values
    ax.set(xticks=x, xticklabels=["Parent", "Metabolite"], xlim=(-0.65, 1.65), ylim=(0, 1.23 * len(rows)),
           ylabel="Relationship rows")
    ax.legend(loc="upper center", ncols=3)
    grid(ax)
    return fig


def ranking_figure(evaluation, data_id):
    policies = evaluation["fixed_budget"]["policies"]
    fig, (recovery, coverage) = figure("Does ranking relative to the parent help?",
                                      "Fixed-budget retrospective evaluation on reserved families.",
                                      f"Source  evaluation.json  •  Counts use observed annotations. Unknown labels remain visible."
                                      f"  •  Dataset {data_id[:8]}")
    for key, label, color, marker, style in zip(POLICIES, POLICY_LABELS, POLICY_COLORS,
                                                 ("o", "s", "^"), ("-", "--", ":")):
        rows = policies[key]
        recovery.plot([row["budget"] for row in rows], [row["bbb"]["positive"] for row in rows],
                      marker=marker, linestyle=style, color=color, linewidth=2, label=label, alpha=0.9)
    recovery.set(title="BBB-positive metabolites recovered", xlabel="Assay budget (candidate slots)",
                 ylabel="Known BBB-positive picks", xticks=[row["budget"] for row in policies[POLICIES[0]]], ylim=(0, None))
    recovery.yaxis.set_major_locator(MaxNLocator(integer=True))
    recovery.legend(loc="upper left")
    grid(recovery)
    budget = policies[POLICIES[0]][0]["budget"]
    at_budget = [next(row for row in policies[key] if row["budget"] == budget) for key in POLICIES]
    y = np.arange(len(at_budget))
    left = np.zeros(len(at_budget))
    for field, label, color in (("positive", "BBB-positive", TEAL), ("negative", "BBB-negative", RED),
                                ("unknown", "Unknown BBB label", GRAY)):
        values = [row["bbb"][field] for row in at_budget]
        bars = coverage.barh(y, values, left=left, height=0.56, color=color, label=label)
        coverage.bar_label(bars, labels=[str(value) if value else "" for value in values], label_type="center",
                           color=INK if field == "unknown" else "white", fontweight="bold")
        left += values
    coverage.set(title=f"Label coverage in the first {budget} picks", yticks=y, yticklabels=POLICY_LABELS,
                 xlim=(0, budget), xlabel="Candidates selected")
    coverage.xaxis.set_major_locator(MaxNLocator(integer=True))
    coverage.invert_yaxis()
    coverage.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2, fontsize=8.5)
    return fig


def endpoint_figure(evaluation, data_id):
    fig, axes = figure("Endpoint performance on held-out data",
                       "Each bar summarizes one endpoint on one cohort.",
                       f"Source  evaluation.json  •  n gives the number of labelled molecules.  •  Dataset {data_id[:8]}")
    cohorts = ("endpoint_test", "held_out_metabolite_annotations")
    for ax, endpoint, name in zip(axes, ENDPOINTS, ENDPOINT_LABELS):
        metrics = [evaluation["endpoint_metrics"][endpoint][cohort] for cohort in cohorts]
        x = np.arange(2)
        for field, label, color, offset in (("roc_auc", "ROC-AUC", BLUE, -0.16), ("ap", "Average precision", TEAL, 0.16)):
            bars = ax.bar(x + offset, [m[field] for m in metrics], width=0.28, color=color, label=label)
            ax.bar_label(bars, fmt="%.3f", padding=3, color=INK, fontsize=9)
        ax.set(title=name, xticks=x, xticklabels=[f"Endpoint test\nn = {metrics[0]['n']}",
                                                f"Reserved metabolites\nn = {metrics[1]['n']}"],
               ylim=(0, 1.17), ylabel="Metric value")
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.legend(loc="upper center", ncols=2, fontsize=9)
        grid(ax)
    return fig


def pgp_figure(comparison, data_id):
    fig, axes = figure("P-gp model trade-offs after the BBB gate",
                       "The same gated molecules are compared within each cohort.",
                       f"Source  pgp_comparison.json  •  Both cutoffs targeted 90% sensitivity on calibration."
                       f"  •  Dataset {data_id[:8]}")
    cohorts = ("endpoint_test_after_bbb", "reserved_metabolite_annotations_after_bbb")
    for ax, cohort, title in zip(axes, cohorts, ("Endpoint test", "Reserved metabolites")):
        results = comparison["cohorts"][cohort]
        x = np.arange(2)
        for key, label, color, offset in (("gnn", "GINE ensemble", BLUE, -0.16),
                                          ("maccs_xgboost", "MACCS XGBoost", AMBER, 0.16)):
            row = results[key]
            counts = (row["tp"], row["fp"])
            totals = (row["positive"], row["n"] - row["positive"])
            bars = ax.bar(x + offset, [k / n for k, n in zip(counts, totals)], width=0.29, color=color, label=label)
            ax.bar_label(bars, labels=[f"{k}/{n}" for k, n in zip(counts, totals)], padding=4, color=INK)
        ax.set(title=f"{title} · n = {results['gnn']['n']}", xticks=x,
               xticklabels=["Substrates\ndetected", "Nonsubstrates\nfalsely flagged"],
               ylabel="Share of true class", ylim=(0, 1.20))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="upper center", ncols=2, fontsize=8.5)
        grid(ax)
    return fig


def benchmark_figure(benchmark, data_id):
    fig, (effect, stability) = figure("Do top-ranked bonds matter to held-out seeds?",
                                     "Top-bond masks are compared with 100 same-size random masks per rotation.",
                                     f"Source  explanations/benchmark.json  •  Paired group-bootstrap 95% intervals."
                                     f"  •  Dataset {data_id[:8]}")
    results = [benchmark["endpoints"][endpoint] for endpoint in ENDPOINTS]
    x = np.arange(len(results))
    for i, row in enumerate(results):
        if row["effect_over_random"] is None:
            effect.text(i, 0, "Unavailable", color=MUTED, ha="center", va="bottom")
            continue
        value = 100 * row["effect_over_random"]
        interval = row["paired_group_bootstrap_95_ci"]
        if interval is not None:
            low, high = (100 * bound for bound in interval)
            effect.vlines(i, low, high, color=TEAL, linewidth=2)
            effect.hlines([low, high], i - 0.032, i + 0.032, color=TEAL)
        effect.plot(i, value, "o", color=TEAL)
        effect.annotate(f"{value:+.2f} pp", (i, value), xytext=(12, 1), textcoords="offset points", va="center")
    effect.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    effect.set(title="Mean effect above random masking", xlim=(-0.45, 1.55), ylabel="Probability change (percentage points)",
               xticks=x, xticklabels=[f"{label}\n{row['eligible_groups']} groups" for label, row in zip(ENDPOINT_LABELS, results)])
    grid(effect)
    for field, label, color, offset in (("seed_top_bond_overlap", "Top-bond overlap", BLUE, -0.16),
                                        ("directional_agreement", "Directional agreement", TEAL, 0.16)):
        bars = stability.bar(x + offset, [row[field] or 0 for row in results], width=0.29, label=label, color=color)
        stability.bar_label(bars, labels=[f"{row[field]:.0%}" if row[field] is not None else "Unavailable"
                                         for row in results], padding=4, color=INK)
    stability.set(title="Agreement across seeds", ylim=(0, 1.2), xticks=x, xticklabels=ENDPOINT_LABELS,
                  ylabel="Mean agreement")
    stability.set_yticks(np.linspace(0, 1, 6))
    stability.yaxis.set_major_formatter(PercentFormatter(1))
    stability.legend(loc="upper center", ncols=2, fontsize=8.5)
    grid(stability)
    return fig


def highlighted_molecule(smiles, effects):
    mol = Chem.MolFromSmiles(smiles)
    highlight = {}
    for row in effects:
        index = row["bond_index"]
        bond = mol.GetBondWithIdx(index)
        if [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] != row["atom_indices"]:
            raise ValueError("Cannot match a saved bond effect to the drawing's atom order.")
        bond.SetProp("bondNote", str(index))
        highlight[index] = (0.52, 0.71, 0.87) if row["mean_effect"] > 0 else (0.90, 0.57, 0.59)
    drawer = rdMolDraw2D.MolDraw2DCairo(920, 345)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.bondLineWidth = 2.5
    options.highlightBondWidthMultiplier = 11
    drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=list(highlight), highlightBondColors=highlight)
    drawer.FinishDrawing()
    return plt.imread(BytesIO(drawer.GetDrawingText()), format="png")


def pair_figure(pair, data_id):
    parent = pair["molecules"]["parent"]["support"]["value"]
    metabolite = pair["molecules"]["metabolite"]["support"]["value"]
    relative = pair["ranking"]["parent_relative_support"]
    support = f"Support   parent {parent:.4f}   ·   metabolite {metabolite:.4f}   ·   relative {relative:+.4f}"
    fig, axes = figure("Why this metabolite advanced",
                       support + "\nUp to three strongest effects per panel. Blue is positive and red is negative.",
                       f"Source  worked-pair explanation.json  •  Effects are original minus masked probability."
                       f"  •  Dataset {data_id[:8]}", rows=2, size=(10.6, 7.5))
    for ax, (endpoint, name, role) in zip(axes, ((ep, label, role) for ep, label in zip(ENDPOINTS, ENDPOINT_LABELS)
                                                for role in ("parent", "metabolite"))):
        molecule = pair["molecules"][role]
        evidence = molecule["endpoints"][endpoint]
        ax.set_axis_off()
        if evidence["status"] == "skipped":
            ax.set_title(f"{role.title()} · {name}", loc="left")
            ax.text(0.5, 0.54, "P-gp skipped", transform=ax.transAxes, ha="center", fontsize=17, fontweight="bold")
            ax.text(0.5, 0.38, "The parent stopped at the BBB gate.", transform=ax.transAxes, ha="center", color=MUTED)
            continue
        decision = {"pass": "pass", "bbb_rejected": "rejected", "pgp_flagged": "flagged"}[evidence["decision"]]
        ax.set_title(f"{role.title()} · {name} {decision}  ({evidence['probability']:.4f})", loc="left")
        effects = [row for row in evidence["bond_effects"] if row["mean_effect"]][:3]
        drawing = ax.inset_axes([0, 0.17, 1, 0.74])
        drawing.set_axis_off()
        drawing.imshow(highlighted_molecule(molecule["canonical_smiles"], effects), interpolation="lanczos")
        strongest = effects[0] if effects else None
        if strongest:
            precision = 3 if abs(strongest["mean_effect"]) >= 0.001 else 5
            ax.text(0.02, 0.04, f"Bond {strongest['bond_index']}   Δ {strongest['mean_effect']:+.{precision}f}"
                    f"   ·   {strongest['directional_agreement']:.0%} sign agreement",
                    color=MUTED, transform=ax.transAxes, fontsize=9)
        ax.set(xlim=(0, 1), ylim=(0, 1))
    return fig


def pilot_figure(pilot, data_id):
    eligible = sorted((pair for pair in pilot["pairs"] if not pair["model_development_overlap"]),
                      key=lambda pair: pair["median_measured_fold_vs_parent"])
    fig, (ax,) = figure("The measured pilot exposes BBB misses",
                        f"{len(eligible)} eligible pairs in {pilot['eligible_groups']} groups. Each dot is a within-study comparison.",
                        f"Source  paired_validation.json  •  Study assays and sampling schedules differ."
                        f"  •  Dataset {data_id[:8]}", columns=1, size=(9.8, 5.3))
    for y, pair in enumerate(eligible):
        fold = pair["median_measured_fold_vs_parent"]
        color = STAGE_COLORS[pair["metabolite_stage"]]
        ax.hlines(y, min(1, fold), max(1, fold), color=color, linewidth=2, alpha=0.65)
        ax.scatter(fold, y, s=64, color=color, zorder=3)
        ax.annotate(f"{fold:.2f}×", (fold, y), xytext=(8, 6), textcoords="offset points", color=INK, fontsize=9)
    ax.axvline(1, color=MUTED, linewidth=1, linestyle="--")
    ax.set(xscale="log", yticks=np.arange(len(eligible)), yticklabels=[pair["metabolite"] for pair in eligible],
           xlabel="Measured metabolite-to-parent fold of the brain-to-systemic ratio (log scale)")
    stage_legend = [Line2D([], [], marker="o", linestyle="", color=STAGE_COLORS[stage], label=label)
                    for stage, label in zip(STAGES, STAGE_LABELS)
                    if any(pair["metabolite_stage"] == stage for pair in eligible)]
    ax.legend(handles=stage_legend, loc="upper left", title="Metabolite screen outcome", fontsize=9)
    ax.set_ylim(-0.6, len(eligible) + 0.25)
    grid(ax, "x")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    data_id, screening, evaluation, comparison, pilot, benchmark, pair = load_reports(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("screening_outcomes", screening_figure, screening),
        ("ranking_policy_comparison", ranking_figure, evaluation),
        ("held_out_model_performance", endpoint_figure, evaluation),
        ("pgp_after_bbb_comparison", pgp_figure, comparison),
        ("bond_masking_benchmark", benchmark_figure, benchmark),
        ("worked_pair_bond_maps", pair_figure, pair),
        ("paired_measurement_pilot", pilot_figure, pilot),
    ]
    for stem, render, report in jobs:
        save(render(report, data_id), args.output, stem, data_id)
        print(args.output / f"{stem}.png")


if __name__ == "__main__":
    main()
