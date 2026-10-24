# Research Log

The decision record for this project: what was investigated, what the literature
already covers, what was built, and what was found. Kept separate from `README.md`
(how to run it) and `docs/ARCHITECTURE.md` (how it works).

Numbers quoted here were correct when written. `README.md` and
`RESEARCH_PAPER_MANUSCRIPT.md` are the current source of truth, and
`B3DB Based Model/scripts/verify_doc_consistency.py` enforces that they match the
artifacts on disk.

## Contents
1. [Novelty Assessment](#part-1--novelty-assessment) — what is and is not publishable, with prior art
2. [Publication Strategy](#part-2--publication-strategy) — the competitive landscape and the recommended paper
3. [Multi-Task GNN Build Log](#part-3--multi-task-gnn-build-log) — the dual-barrier GNN, its null result, and the metric suite
4. [Prior Audit Disposition](#part-4--prior-audit-disposition) — an earlier code audit and how each finding was resolved

---


# Part 1 — Novelty Assessment

**Prepared:** 2026-10-24
**Scope:** audit of `RESEARCH_PAPER_MANUSCRIPT.md` against current literature, plus a measured
result that changes what this project can claim.
**Status:** analysis only. No code was modified, no experiment was run, nothing was committed.

---

### 1. Verdict

**As currently framed, the manuscript has no defensible novelty claim.** Five of its seven stated
contributions are already published, and two are contradicted by tables inside the manuscript
itself. Submitting it to JCIM in its present form invites a desk reject, and the most likely
reviewer objection is one the draft cannot answer: the closest prior art is a 2004 JCIM paper whose
PDF sits in this repository, uncited.

**However, one measurement taken during this audit appears to be unpublished and is strong enough to
carry a paper.** On the 216 molecules where the two benchmarks overlap, a BBB permeability label
tells you essentially nothing about whether a molecule is actively effluxed, and roughly **a third
of molecules the benchmark calls "permeant" are confirmed P-glycoprotein substrates**. The
literature states that passive permeability and efflux are mechanistically distinct, but no source I
found quantifies this at the label level or draws out what it means for benchmark validity.

That reframes the project from *"we built a dual-barrier tool"* (scooped in 2004) to *"we quantified
a blind spot in the field's standard benchmark"* (novel, falsifiable, and useful to every group
training on B3DB or BBBP).

**Recommendation: two papers.** A methods paper led by the independence result, and a separate
translational paper carrying the pharmacopeia screen and the cryo-EM docking. Neither then has to
carry the other's weaknesses.

---

### 2. What will not survive review

| Manuscript claim | Status | Prior art / evidence |
|---|---|---|
| "Unified dual-barrier framework" (contributions 1, 5) | **Scooped, 2004** | Adenot & Lahana, *J. Chem. Inf. Comput. Sci.* **2004**, 44, 239-248 built a BBB model *and* a P-gp substrate model and proposed a combined CNS map explicitly including P-gp substrates. **The PDF is in this repo** at `Initial Model/reference_papers/adenot_2004_jcim.pdf`, and the paper is **absent from the reference list.** |
| "via 3D Conformational Modeling" (title, contribution 2) | **Scooped *and* self-refuted** | Tiwari et al., *Pharmaceutics* **2026**, 18, 1018 reached the same negative conclusion on B3DB. Worse, `results/ablation_study_results.csv` shows Morgan+2D at ROC-AUC **0.9509** beating the deployed Morgan+2D+3D at **0.9481**, and 2D-only 0.9185 vs 2D+3D 0.9192 (+0.0007). The title claims a benefit the project's own table refutes. |
| TreeSHAP + Morgan bit deconvolution; TPSA as top BBB feature (contributions 3, 4) | **Scooped** | Fingerprint-based explainable BBB ML, bioRxiv **2025**.10.25.684495. Tiwari et al. 2026 also report TopoPSA as the top SHAP feature, tied to desolvation: the same finding and the same interpretation. |
| Scaffold-split robustness (section 4.1) | **Scooped, and spun backwards** | *J. Chem. Inf. Model.* **2026**, 66, 6837 ("Overrepresentation Bias..."); Guo et al. **2024**, "Scaffold Splits Overestimate Virtual Screening Performance". Section 4.1 says the split "prov[es] substantial generalizability" while its own Table 1 shows specificity **0.848 to 0.580** and MCC **0.754 to 0.572**. Reviewers read the table. |
| ROC-AUC 0.9522 on B3DB | **Mid-pack** | Published B3DB results cluster at 0.93-0.95; a 2025 geometric GNN reports 0.9685 *under scaffold splits* (arXiv 2507.18926). This cannot be a performance paper. |
| Applicability domain, SMARTS counterfactuals, quadrant matrix (contribution 5) | **Standard toolkit** | Competent engineering, not contributions. |

#### What *is* defensible

- The **216-molecule label-independence measurement** (section 3): the lead claim.
- The **scaffold-split asymmetry** between the two tasks (section 3.3): supporting, and not
  something I found reported.
- **ML efflux probability vs. cryo-EM docking dG in 6QEX**: no prior study correlates these. The
  closest (Kadioglu & Efferth, PMC6829872) compared docking *poses*, not energies. This needs the
  fix in defect A2 before it can be claimed.
- The **3D-descriptor negative result**, reframed as independent replication of Tiwari et al. 2026
  on the *classification* split (7,805 compounds, XGBoost) rather than their *regression* subset
  (1,058 compounds, LightGBM). Confirmatory, not novel, but publishable as a supporting section.

---

### 3. The novel result

#### 3.1 The overlap set

B3DB and Broccatelli share **216 molecules with unambiguous ground truth on both axes**. Matching
was by full InChIKey; the count reproduces identically from the raw and the processed files, and a
stricter canonical-SMILES equality test gives 214. All 216 carry a single unanimous label per side,
so nothing was excluded as ambiguous.

|  | P-gp substrate | P-gp non-substrate | total |
|---|:---:|:---:|:---:|
| **BBB+ (permeant)** | 45 | 95 | **140** |
| **BBB- (non-permeant)** | 19 | 57 | **76** |
| **total** | **64** | **152** | **216** |

A looser connectivity-layer match gives 343 blocks, but 40 of those contain B3DB stereo-variants
that *disagree* on the BBB label, so only 303 are usable. Report 216; mention 343 as an upper bound
only. (B3DB enumerates stereoisomers heavily: 7,805 distinct molecules collapse to 4,022
connectivity blocks, and 130 blocks carry conflicting BBB labels internally. That is itself worth a
sentence in the paper, as it affects anyone deduplicating B3DB.)

#### 3.2 The claim, stated at two strengths

**Primary claim (robust: a prevalence, not a hypothesis test):**

> **32.1% of B3DB BBB+ molecules in this set are confirmed P-glycoprotein substrates**
> (45/140, 95% CI [24.5%, 40.6%], Clopper-Pearson).

This does not depend on any significance test. Roughly one in three molecules a BBB classifier is
rewarded for calling "permeant" is a molecule whose brain exposure is actively opposed, and the
benchmark records no credit or penalty for getting that right or wrong. **A model can top the B3DB
leaderboard while being systematically wrong about net brain exposure for a third of its true
positives.**

**Secondary claim (an association test: state its limits honestly):**

> No detectable association between the two labels: chi-squared = 0.887, **p = 0.346**; Fisher
> OR = **1.42, 95% CI [0.76, 2.67]**; Cramer's V = **0.064**. Risk difference 7.1 pp, 95% CI
> [-5.3, +19.6]. P(P-gp+ | BBB+) = 32.1% vs P(P-gp+ | BBB-) = 25.0%.

**Do not write "the labels are statistically independent."** The data are *consistent* with
independence and the shared variance is negligible (V = 0.064), but at n = 216 the confidence
interval still admits an odds ratio up to about 2.7. The correct phrasing is that the BBB label is
**not a usable proxy for efflux status** (you cannot infer one from the other) and that any residual
association is bounded as weak. State the power limitation in the paper; a reviewer will ask.

#### 3.3 Supporting finding: the two axes generalize differently

From `results/validation_metrics_summary.csv`, going from the random split to the Bemis-Murcko
scaffold split:

| | ROC-AUC random | ROC-AUC scaffold | change | Specificity random to scaffold |
|---|:---:|:---:|:---:|:---:|
| **BBB** | 0.9522 | 0.9048 | **-0.047** | 0.848 to **0.580** |
| **P-gp** | 0.9587 | 0.9631 | **+0.004** | 0.919 to 0.980 |

The efflux axis transfers to unseen chemotypes; the permeability axis does not. Under the framing
above this is coherent rather than puzzling: P-gp substrate recognition is driven by a compact set
of physicochemical and shape features that generalize, whereas the BBB label is a composite endpoint
whose non-permeant class is held together partly by chemotype.

It also means the BBB model's out-of-domain failure mode is specifically **calling almost everything
permeant** (specificity 0.580), which is exactly the region where section 3.2 says efflux is
unaccounted for. The specificity collapse the current draft tries to explain away becomes an asset.

**Caveat that must be stated:** `scripts/run_validation.py` trains a **fresh, untuned** model for
the scaffold evaluation, so this is not an apples-to-apples comparison with the Optuna-tuned
random-split model. Either re-tune within the scaffold protocol or label the comparison explicitly.

#### 3.4 A second cross-labelled source (weak, caveated)

The Adenot sheet of `Initial Model/Dataset/raw_adenot_li.xls` has a `P-gp` column. It is **not a
label**: it holds provenance strings ("P-gp (Adenot, J. Chem. Inf. Comput. Sci. 2004, 44,
239-248)") marking 88 of 1,632 rows as drawn from P-gp substrate compilations. Usable only as an
88-molecule supplementary check, and only if the semantics are stated precisely.

Worth noting in passing: `Initial Model/clean_raw_data.py` selects
`["SMILES", "SMILES-B", "Name", "BBB-crossing"]` and discards this column, so the pipeline throws
away its only other cross-labelled annotation.

---

### 4. Paper 1: methods

**Target:** *Journal of Cheminformatics* (or JCIM). Both publish benchmark-critique work, which is
what this now is.

**Working title:** *"Blood-Brain Barrier Benchmarks Are Blind to Active Efflux: Quantifying What
Single-Axis Permeability Labels Omit"*

**Thesis:** the standard BBB label is not a proxy for efflux status, so a single-axis BBB benchmark
cannot measure net brain exposure. We quantify how much it omits and show the omission is
concentrated where models are already weakest.

**Abstract skeleton:**
1. BBB classification benchmarks (B3DB, BBBP) are the field's standard target; reported ROC-AUC now
   exceeds 0.95.
2. But the labelled endpoint is composite, and these benchmarks carry no efflux annotation.
3. Using the 216-molecule intersection of B3DB and Broccatelli, we show a BBB label is not a usable
   proxy for P-gp substrate status (V = 0.064; OR 1.42 [0.76, 2.67]), and that 32.1%
   [24.5-40.6%] of BBB+ molecules are confirmed substrates.
4. Out-of-fold error analysis across all 7,805 B3DB molecules locates where this matters (E1).
5. The two axes generalize differently: scaffold-split transfer costs BBB -0.047 ROC-AUC and
   collapses specificity to 0.580, while P-gp is unaffected.
6. We replicate, on the classification split, the finding that 3D conformer descriptors add nothing
   over 2D plus fingerprints.
7. Recommendation: report efflux-stratified metrics alongside BBB accuracy; we release the
   216-molecule cross-labelled set as an evaluation resource.

**Figures:** (1) contingency table plus independence statistics with CIs; (2) E1 efflux-stratified
error analysis; (3) scaffold degradation, BBB vs P-gp; (4) corrected cross-task SHAP comparison.

**Releasable artifact:** the 216-molecule dual-labelled set. Small, concrete, immediately reusable:
this is often what gets a benchmark-critique paper cited.

#### E1: efflux-stratified error analysis (specification; not run)

**Why a new run is unavoidable.** No per-molecule predictions exist anywhere in the repository (all
13 CSVs were checked), and `run_validation.py` computes `probs`/`preds` in memory and persists only
the aggregated 10-row metric table plus confusion-matrix PNGs. The 216 overlap molecules also sit
inside B3DB's training partition, so in-sample predictions are meaningless.

**Design:**
- Generate **out-of-fold** P(BBB+) for all 7,805 molecules via stratified K-fold. This requires
  **retraining K models**; it is not inference-only. XGBoost on 7,805 x 2,054 is minutes.
- Refit the scaler **inside each fold**. Reuse the within-fold pattern already implemented correctly
  in `scripts/run_ablation_study.py`; do not copy the global-scaling pattern used elsewhere.
- **Primary test (ground-truth anchored, n = 216):** among **BBB-** molecules, compare the
  false-positive rate for P-gp substrates (n = 19) against non-substrates (n = 57). This is the
  "efflux masquerades as a false positive" hypothesis.
- **Secondary test (full scale, n = 7,805):** regress out-of-fold BBB *probability* (continuous, far
  better powered than binary error) on predicted P(P-gp+), restricted to the P-gp model's
  applicability domain via `explainability/applicability_domain.py`. Note the circularity (one model
  explaining another's errors) and anchor it on the 216 ground-truth cases.

**Plan for a null result.** The n = 19 cell is underpowered, and section 3.2's near-zero association
already suggests E1 may come back null. That is survivable and must be planned for: **a null E1
reinforces the thesis**, because if efflux status does not predict BBB-model error then the BBB label
is not efflux-informed, which is the paper's point. Write the analysis both ways before running it,
and do not draft the conclusion first.

---

### 5. Paper 2: translational

**Target:** *Molecular Pharmaceutics* or *Pharmaceutics*.

**Working title:** *"Two-Axis Screening of the FDA-Approved Pharmacopeia Identifies Latent CNS
Candidates and Efflux-Limited Neuropsychiatric Drugs, with Cryo-EM Structural Corroboration"*

**Assets, all already on disk, no re-run needed except E2:**
- 2,405-compound screen; quadrant split Q1 710 (29.5%) / Q2 240 (10.0%) / Q3 1,064 (44.2%) /
  Q4 391 (16.3%).
- 273 non-CNS Q1 repurposing leads (`top_cns_repurposing_candidates.csv`).
- 105 Q2 neuropsychiatric drugs (`cns_drugs_efflux_resistance_profiles.csv`).
- 6QEX docking, 21 compounds: Q2 -8.94 +/- 1.12 vs Q1 -6.83 +/- 1.37 kcal/mol, Welch t = -3.530,
  p = 0.0035; Mann-Whitney U = 11.0, p = 0.0043.
- Benchmark case studies (Loperamide Q2, Levetiracetam Q1).

**Figures:** (1) pharmacopeia two-axis landscape; (2) disease-area distribution; (3) docking energy
distribution by quadrant; (4) corrected ML-vs-dG correlation with outliers labelled.

#### E2: re-run the docking correlation with live model output (specification; not run)

`scripts/run_cryoem_docking_validation.py` hard-codes `prob_pgp_ml` as 22 literal constants (first
at line 66). So section 6.4's "cross-modality correlation" is docking energies against numbers typed
into a script. The arithmetic *is* reproducible from the saved CSV (I recomputed rho = -0.5965,
p = 0.004315), but nothing in the pipeline ties that column to the model, and a reviewer who opens
the script will see it. Replace the literals with a live call to the saved booster through
`core/features.prepare_dataframe` plus `pgp_scaler_metadata.json`.

#### Honesty requirements for Paper 2

These are not optional polish; each is a point a reviewer will otherwise raise.

1. **Report the correlation over all compounds, not only Q1 vs Q2.** The current test excludes the
   three Q3 compounds, and one of them is the problem: see the next item.
2. **Confront the outliers instead of omitting them.** Penicillin G (P(P-gp+) = 0.034) docks at
   **-9.00 kcal/mol** and Betamethasone (0.015) at **-9.22**, both tighter than seven of the eight
   Q1 compounds and tighter than co-crystallised Paclitaxel (-8.57). **Docking dG alone therefore
   does not classify substrates.** This is consistent with published findings that P-gp docking
   cannot discriminate substrate from non-substrate, and it is an argument *for* a ligand-topology
   model rather than against it. Handled openly it becomes the contribution (nobody has correlated
   ML efflux probability with dG in 6QEX) rather than the hole in the paper.
3. **Tone down the per-drug clinical assertions.** "Betamethasone for vasogenic cerebral edema"
   presents as discovery something for which dexamethasone is already standard of care; one such
   claim invites doubt about the other 272. Present the screen as hypothesis-generating and let the
   table speak.
4. **Disclose the applicability-domain weakness.** Of 2,405 screened compounds, **1,454 are only
   "Borderline" and 0 are "Outside"**: an AD gate that rejects nothing is not evidence of coverage.
   This caps how strongly any individual compound can be claimed.
5. **Justify the score.** See defect E14 below.

---

### 6. Defect register

#### A. Invalidating: these results are currently wrong

**A1. Global SHAP is computed on unscaled inputs.** *(RESOLVED 2026-10-24: `core.features.load_processed_dataset` now applies the training-derived scaler and raises if metadata is missing; the split is corrected to `test_size=0.15, stratify=y`. Recomputed: TPSA 0.9875 -> 1.1825, MolWt 0.6955 -> 0.5366. The TPSA-vs-LogP conclusion survives.)*
`scripts/run_explainability_analysis.py` applies no scaler before explaining a model trained on
z-scored descriptors. **Every number in manuscript Table 3 is therefore invalid**, including the
headline "TPSA dominates BBB, LogP dominates P-gp" and the rho = 0.7714 cross-task correlation. The
symptom is visible: MolWt enters at about 384 (its raw mean) where a correctly scaled feature would
enter near 0.
*Root cause:* the processed CSVs store **raw** descriptors by design, so every consumer must scale
independently, and 3 of 10 call sites forget. Fix forward by adding
`core/features.load_processed_dataset(path, is_pgp)` that returns scaled features, and route all
call sites through it so the invariant lives in one place.
*Also in that script:* the BBB split is `test_size=0.2` unstratified, while its own comment claims
it matches the training split (0.15, stratified).

**A2. Docking ML probabilities are hard-coded literals.** *(RESOLVED 2026-10-24: all 23 literals replaced with live inference; quadrants derived from the predictions. This exposed that six benchmark SMILES had the wrong molecular skeleton, now resolved from the curated library, and that AutoDock Vina was never seeded - fixed with VINA_SEED = 42. Re-docked: Welch t = -3.7933, p = 1.60e-03; Spearman rho = -0.6117, p = 2.49e-03.)*

**A3. Two legacy evaluation scripts feed unscaled features to scaled boosters.** *(RESOLVED 2026-10-24: both scripts and their two tracked figures deleted.)*
`BBB/bbb_model_evaluation.py` and `Pgp/pgp_model_evaluation.py` both pass raw `X` to a model trained
on z-scored input, and `bbb_model_evaluation.py` additionally uses an 80/20 *unstratified* split.
Their metrics are invalid and two figures derived from them are still tracked. Delete both, or fix
and regenerate.

#### B. Internal contradictions a reviewer will catch

**B4. The abstract contradicts Table 1 of the same document.**

| Metric | Abstract says | `validation_metrics_summary.csv` and Table 1 say |
|---|---|---|
| BBB ROC-AUC | 0.9544 [0.9449-0.9644] | **0.9522 [0.9410-0.9629]** |
| BBB Accuracy | 0.8962 [0.8815-0.9116] | **0.8856 [0.8659-0.9035]** |
| P-gp ROC-AUC | 0.9593 [0.9320-0.9822] | **0.9587 [0.9330-0.9806]** |

The abstract carries stale numbers from an earlier run, preserved in
`B3DB Based Model/docs/SCIENTIFIC_DOCUMENTATION.md`. Quote only values that trace to a current
artifact.

**B5. All six rows of Table 3 disagree with the saved CSV.** Manuscript vs
`shared_properties_quantitative_comparison.csv`: TPSA 1.0775 vs **0.9875**; MolWt 0.7107 vs
**0.6955**; LogP 0.4087 vs **0.3957**; HDonors 0.2962 vs **0.2548**; HAcceptors 0.1063 vs
**0.1118**; RotBonds 0.1653 vs **0.1591**. Doubly wrong: stale *and* computed unscaled (A1).

**B6. Topotecan is listed but was never docked, and the reported p-value proves it.**
Section 6.1 says "Twenty-two representative therapeutics" and lists Topotecan in Q1, but Table 7 has
21 rows and no Topotecan. Verified root cause: the SMILES at
`run_cryoem_docking_validation.py:185` (`CCN(C)Cc1cc2c(cc1O)nc3n(c2=O)CC4(O)C(=O)OCc34`) **is
chemically invalid**; RDKit fails to kekulize it ("Unkekulized atoms: 5 6 7 8 9 10 12 13 24") and
`MolFromSmiles` returns `None`. `prepare_ligand_pdbqt` then returns `False` at its `if mol is None`
guard, *before* the `except` block that would have printed a message, so the compound is dropped in
silence. No Topotecan file exists in `data/docking/`. Independently confirmed by the statistics: the
manuscript's reported rho p-value of 0.0043 corresponds to **n = 21** (n = 22 gives 0.0034).
*Fix:* use a valid Topotecan SMILES and re-dock, or remove the compound from the manuscript. Also
make `prepare_ligand_pdbqt` log the `mol is None` path.

#### C. Citation and data errors

**C7. B3DB is miscited.** Section 3.1 and reference 2 attribute it to "Martins et al., 2012", which
is a different dataset (a Bayesian BBB model). B3DB is *"A curated diverse molecular database of
blood-brain barrier permeability with chemical descriptors"*, Meng et al., *Scientific Data*
**2021**. This misattributes the paper's primary data source; verify the full author list and DOI
against the record before citing.

**C8. Adenot & Lahana 2004 is uncited.** The closest prior art to the central claim, with the PDF in
this repository. Must be cited *and* explicitly distinguished: the honest distinction is that they
proposed the combined map, while this work quantifies why one is necessary.

**C9. P-gp class balance is wrong.** Section 3.1 states 668 substrates / 537 non-substrates
(55.4% / 44.6%). Measured from `processed_pgp_data.csv`: **650 / 555 (53.94% positive)**. The raw
`pgp_broccatelli.tab` is 651 / 568 = 1,219 before the 3D-conformer gate drops 14 molecules. The
manuscript's total (1,205) is right but its split is not.

#### D. Statistical reporting

**D10. A non-significant correlation is presented as a finding.** Section 4.3 reports rho = 0.7714
as the "Cross-Task Correlation" supporting the shared-property argument, but **p = 0.0724** at
n = 6. State it as non-significant, or drop the claim. (The value itself reproduces exactly from the
current CSV.)

**D11. Checked and fine, no action.** Section 6.2's Mann-Whitney p = 0.0043 matching the Spearman p
is a rounding coincidence, not a copy error: recomputed, MWU = 0.004274 and Spearman = 0.004315, and
both round to 0.0043.

#### E. Framing

**E12. Remove "via 3D Conformational Modeling" from the title.** The project's own ablation refutes
it. Reframe the 3D descriptors as a replicated negative result citing Tiwari et al. 2026, and note
the real cost: the ETKDGv3 embedding gate drops 14 of 1,219 molecules (1.1%) to buy nothing.

**E13. Rewrite section 4.1.** "Proving substantial generalizability" contradicts the table beneath
it. Under the new framing the specificity collapse (0.848 to 0.580) is an asset, not an
embarrassment: it shows the model defaults to "permeant" out of domain, precisely where section 3.2
says efflux goes unrecorded.

**E14. Justify S_CNS, or it reads as double-counting.** `S_CNS = P(BBB+) * (1 - P(P-gp+))`
multiplies two probabilities as if independent. If the BBB label already reflects efflux, and logBB
is a composite endpoint so in general it does, the product double-counts. **Section 3.2's result is
what licenses the multiplication**: the two labels carry near-independent information, so the
factorization is defensible. Say this explicitly. This is also the cleanest intellectual link from
Paper 1 to Paper 2.

#### Out of scope for both papers

The GNN in `Initial Model/` should appear in neither manuscript. `train_gnn.py` contains
`if graph.get("source", "") == "Li": continue`, silently discarding all 269 Li molecules from a
1,890-molecule corpus (14.2%), so every reported metric is Adenot-only on about 146 test molecules
per fold. And **no GNN metrics are persisted anywhere** (12 `.pt` checkpoints exist, zero result
files), so any number quoted for it would have to come from prose or a re-run.

---

### 7. Asset division

| Existing artifact | Paper 1 | Paper 2 | Note |
|---|:---:|:---:|---|
| 216-molecule cross-labelled set (new) | **lead** | - | Release as supplementary data |
| `validation_metrics_summary.csv` (scaffold asymmetry) | **yes** | context | Re-tune or caveat the scaffold model |
| `ablation_study_results.csv` | **yes** | - | Reframed as replication |
| `shared_properties_quantitative_comparison.csv` | **yes** | - | Blocked on A1 |
| E1 out-of-fold error analysis (to run) | **yes** | - | Core new experiment |
| `fda_repurposing_dual_barrier_screen.csv` (2,405) | - | **lead** | Disclose AD weakness |
| `top_cns_repurposing_candidates.csv` (273) | - | **yes** | Tone down per-drug claims |
| `cns_drugs_efflux_resistance_profiles.csv` (105) | - | **yes** | Strongest translational hook |
| `cryoem_pgp_docking_validation.csv` (21) | - | **yes** | Blocked on A2 and B6 |
| Loperamide / Levetiracetam case studies | - | **yes** | Loperamide is a genuine zero-shot case |
| GNN (`Initial Model/`) | - | - | Excluded; see above |

---

### 8. Sequencing

1. **Fix A1, A2, A3 first.** E1 and the SHAP regeneration must not inherit the scaling bug.
2. **Then run E1** and let its result choose between the two framings prepared in section 4.
3. **Then correct B4-B6, C7-C9, D10** against the regenerated artifacts.
4. **Then split and rewrite** along sections 4 and 5.
5. Re-run `tests/test_explainability.py` (8 four-quadrant contracts) after the
   `load_processed_dataset` refactor.

**One discipline worth keeping:** every number in both manuscripts should trace to a file on disk,
not to prose. Three of the defects above (B4, B5, and the BBB ROC-AUC discrepancy in
`SCIENTIFIC_DOCUMENTATION.md`) exist only because numbers were copied from an earlier run's
write-up instead of re-read from the artifact.

---

### 9. Sources consulted

**Prior art establishing the scoop:**
- Adenot, M.; Lahana, R. *J. Chem. Inf. Comput. Sci.* 2004, 44, 239-248. DOI 10.1021/ci034205d
- Tiwari et al. *Pharmaceutics* 2026, 18, 1018. DOI 10.3390/pharmaceutics18081018 (PMID 42654135)
- "Overrepresentation Bias Leads to Performance Overestimation in BBB Permeability Prediction
  Models." *J. Chem. Inf. Model.* 2026, 66, 6837
- Guo et al. "Scaffold Splits Overestimate Virtual Screening Performance." 2024, arXiv 2406.00873
- "Fingerprint-Based Explainable Machine Learning for Predicting BBB Permeability."
  bioRxiv 2025.10.25.684495
- "Geometric Multi-color Message Passing GNNs for BBB Permeability Prediction." arXiv 2507.18926

**Establishing the gap is open:**
- Kadioglu & Efferth. ML platform for P-gp modulators validated by docking. PMC6829872: compares
  docking *poses*, not energies, to ML predictions.
- Conrad, Paras & Vaz. "Model of P-Glycoprotein Ligand Binding and Validation with Efflux Substrate
  Matched Pairs." *J. Med. Chem.* 2024, 67, 5854.
- Alam et al. *Science* 2019, 363, 753-756: the 6QEX structure.
- Searches for a study quantifying how often P-gp efflux produces BBB false positives, and for any
  ML-efflux-probability vs. 6QEX docking-energy correlation, returned no match.

**Caveat on this section:** these came from literature search, and I read result excerpts rather
than full texts for most. Verify each DOI and the specific numbers before citing them in a
submission.


---


# Part 2 — Publication Strategy

**Prepared:** 2026-10-24
**Repo state at time of writing:** HEAD `e13f3a5 MDs Added`, working tree clean. That commit added
`BLOCK_DIAGRAM.md` and Part 1 only. No Python has changed since
`a53789f`, so every defect listed in Part 1, Defect Register is still live.
**Method:** 14 literature searches plus full-text retrieval of the two most threatening papers.

---

### 1. Headline: the landscape moved, and it moved against the current framing

Part 1 concluded the dual-barrier framing was scooped by Adenot & Lahana (2004).
That was right but incomplete. A deeper sweep found **five more relevant papers from 2025-2026**, the
newest three months old. The "combine BBB prediction with efflux prediction" lane is now thoroughly
occupied:

| Paper | What it did | Why it threatens the current draft |
|---|---|---|
| **BBB-Nuke**, bioRxiv 2026.07.13.738280 (Jul 2026) | Trains on **B3DB (7,807)** + 2 benchmarks = 9,262 compounds. Predicts substrate probability for **7 efflux transporters** (MDR1, BCRP, MRP1, MRP2, MRP4, MATE1, OAT3) and feeds them as features into a gradient-boosted BBB classifier. AUROC 0.933 CV. Ships as pip package, REST API and MCP server. Screened 1B+ compounds. | **This is your architecture, done more comprehensively** (7 transporters vs your 1) on the same dataset, with tooling. Your two-model setup is now a strict subset. |
| **Daood et al.**, *Mol. Pharmaceutics* 2025, 22, 7022 (Oct 2025) | Hand-curated >24,000 bioactivity records from >900 sources into 8 ABC-transporter datasets (~8,800 chemicals). Links predicted efflux to brain exposure on 85 measured Kp,uu compounds. CC-BY 4.0. | Occupies "predicted efflux substrates have lower brain exposure." |
| **JCIM 2024**, 10.1021/acs.jcim.4c02212 | 154 drugs, 3-class model separating **efflux substrates from CNS+ and CNS−** (AUC 0.82). Beats CNS-MPO (0.53), BBB-Score (0.68). | Occupies "efflux deserves its own class, not a merged label." |
| **Dolgikh et al.**, *JCIM* 2016, 56, 2225 | QSAR for Kp,uu,brain **incorporating P-gp efflux as a variable**; ~1,000 compounds; adding P-gp improved test performance (rho 0.73, n=91). | Occupies "efflux improves brain-exposure prediction" — since 2016. |
| **JCIM 2026**, 10.1021/acs.jcim.6c00819 + bioRxiv 2026.02.26.708193 | Audits of TDC ADMET leaderboards: leakage, duplicated compounds with conflicting labels, `bbb_martins` rank 8 to 1 by deliberate overfitting. | Occupies the generic "the benchmark is flawed" critique. |

Also relevant: *Sci. Rep.* 2024 (s41598-024-59734-9) handles the efflux problem by **deleting P-gp
substrates** from its BBB training set "as these are likely to be removed from the CNS."

**Read plainly: a paper whose contribution is "we built a BBB model and a P-gp model and combined
them" is not publishable in 2026.** That is settled. Stop defending it.

---

### 2. But the sweep also found the gap — and it is sharper than before

Two facts, both verified from full text, not snippets:

**Fact 1 — BBB-Nuke never checks its target against reality.** In the full text, `Kp,uu` appears
exactly once, in a figure caption describing a feedback loop the paper never implements. "Unbound"
and "Friden" do not appear at all. Brain-to-plasma ratio appears only as a stated limitation and a
future direction. The paper **treats B3DB as ground truth and never analyses the label** — no
discussion of label definition, threshold, source disagreement or noise.

**Fact 2 — Daood et al. never touch a BBB benchmark.** Their brain-exposure analysis uses 85 Fridén
rat Kp,uu compounds against *predicted* transporter status. B3DB and every other BBB permeability
benchmark are absent from the paper. And their headline "twice or more likely" claim is reported
**with no statistical test and no confidence interval.**

So the field has split into two camps that never meet:

- Camp A (BBB-Nuke, LightBBB, ADMETlab, your repo): optimize accuracy **on the BBB± label**.
- Camp B (Dolgikh, Daood, Loryan): model **actual brain exposure**, on small or proprietary data.

**Nobody has asked whether Camp A's target predicts Camp B's endpoint.** That question is open, it is
cheap to answer with public data, and your measurement from Part 1, The Novel Result is the
reason to suspect the answer is no.

---

### 3. The recommended claim

> **Accuracy on BBB permeability benchmarks does not transfer to net brain exposure, and the
> benchmark label is why.** The BBB± label carries no usable information about active efflux, so
> models that improve on it — including transport-aware ones — cannot be improving prediction of
> brain exposure. We quantify the label's efflux blindness, demonstrate the failure to transfer
> against measured Kp,uu, and release a cross-labelled resource and an exposure-validated protocol.

#### Three legs

**Leg 1 — Label mis-specification (you already have the seed).**
B3DB's BBB± label is uninformative about ground-truth P-gp substrate status: n = 216,
Cramer's V = 0.064, OR 1.42 [0.76, 2.67], p = 0.346; 32.1% [24.5-40.6%] of BBB+ molecules are
confirmed substrates. **This needs scaling up** — see section 5.

**Leg 2 — The failure to transfer (the killer experiment, and the actual novelty).**
Assemble the public measured-Kp,uu compounds and run a head-to-head: how well do BBB-label-trained
predictors recover *measured net brain exposure*? Baselines that are all freely available: your BBB
model, CNS-MPO, BBB-Score, LightBBB, ADMETlab, and **BBB-Nuke itself** (pip-installable, so it costs
you an afternoon to include the current state of the art as a baseline). Then show whether an
explicit efflux axis closes the gap.
*This is the part nobody has done.* Verified: BBB-Nuke has no Kp,uu validation; Daood's version has
no statistics and no BBB benchmark.

**Leg 3 — Resource and protocol.**
Release the cross-labelled set plus a recommendation: report efflux-stratified and
exposure-validated metrics alongside BBB AUC. Dataset-and-protocol papers are exactly what
*Journal of Cheminformatics* and *Scientific Data* publish, and the artifact is what gets cited.

#### Why each competitor cannot claim this

| Competitor | Why it does not block Leg 2 |
|---|---|
| BBB-Nuke 2026 | No Kp,uu validation anywhere; treats B3DB as ground truth; never questions the label |
| Daood 2025 | Never uses B3DB or any BBB benchmark; "2x" claim has no statistical test |
| Dolgikh 2016 | Proprietary in-house data, regression model, not a benchmark critique, not reproducible |
| JCIM 2024 | 154 drugs, 3-class label; no Kp,uu; does not question B3DB |
| TDC audits 2026 | Audit label **correctness** and leakage. This is label **mis-specification** — the label is correct and still measures the wrong thing. Different axis. |
| Adenot 2004 | Built the dual model; never tested label adequacy or brain exposure |

**Bonus finding worth a paragraph of its own.** BBB-Nuke labels a compound a substrate if
**"IC50 is below 10 uM or the efflux ratio is above 2.0."** IC50 is an *inhibition* measure, not
transport. Their seven "substrate" models are therefore partly inhibition models — a real
methodological error in the current state of the art, and exactly the kind of thing a careful
cross-labelled resource exists to prevent. Their external test set also appears in their own
training list (Benchmark2.5 is both "never used in training" and among the 9,262 CV compounds), a
contradiction the preprint does not address. Both are fair, citable observations.

---

### 4. What to abandon

Being blunt, because the sunk cost is the enemy here:

- **The current title and framing.** Dead.
- **3D conformer descriptors as a selling point.** Your own ablation refutes it (`Morgan+2D` 0.9509
  beats `Morgan+2D+3D` 0.9481) and *Pharmaceutics* 2026, 18, 1018 published the same conclusion.
- **Any claim to competitive BBB accuracy.** 0.9522 is mid-pack; BBB-Nuke reports 0.933 CV on a
  larger set and a 2025 GNN reports 0.9685 under scaffold splits.
- **The GNN arm** (`Initial Model/`). Discards 269 Li molecules silently; no metrics persisted.
- **"We built a dual-barrier framework."** Adenot 2004, and now BBB-Nuke with 7 transporters.

### What to keep, repurposed

- **The two trained models** — demoted from "the contribution" to "one baseline among six."
- **The 216-molecule overlap** — promoted from a footnote to the seed of the released resource.
- **The cryo-EM 6QEX docking** — still genuinely unoccupied as an ML-vs-dG correlation, but only
  after the hard-coded probabilities are fixed. Belongs in Paper 2 as mechanism.
- **The FDA pharmacopeia screen** — Paper 2's lead.
- **`core/features.py`** — the shared feature kernel is reusable for the resource build.

---

### 5. Concrete next steps, in order

**Step 0 — fix the invalidating defects first (days).**
A1 (unscaled SHAP), A2 (hard-coded docking probabilities), A3 (two legacy eval scripts) from
Part 1. Everything downstream inherits these bugs otherwise. No Python has changed
since `a53789f`, so none of this is done yet.

**Step 1 — build the cross-labelled resource (1-2 weeks).**
Download Daood et al.'s supporting file `mp5c01065_si_002.xlsx` (458 KB, **CC-BY 4.0**, free from the
ACS article page) — it contains transporter interactions for ~8,800 compounds across eight sets.
Intersect each with B3DB by InChIKey, as already done for Broccatelli.

Realistic expectations, so you plan with the right number. Their *substrate* sets are smaller than
the "8,800" headline suggests, because that figure is dominated by inhibition data:

| Daood substrate set | Size | Split |
|---|:---:|---|
| P-gp | 607 | 297 sub / 310 non-sub |
| BCRP | 507 | 253 / 254 |
| MRP2 | 253 | 134 / 119 |
| MRP1 | 182 | 94 / 88 |

Combined with Broccatelli (1,205, giving 216 B3DB matches), a realistic yield is **roughly 350-600
molecules carrying a BBB label plus at least one transporter label**, some with two or three
transporters. That is 2-3x your current n, enough to tighten the OR confidence interval
substantially, and a **tri-axis (BBB x P-gp x BCRP) resource does not currently exist**. Report the
yield honestly whatever it turns out to be.

**Step 2 — assemble the public Kp,uu evaluation set (1 week).**
- Fridén et al. rat Kp,uu, 85 compounds after dedup (the set Daood used — cite them).
- A Frontiers 2024 compilation (~157 compounds, stated as publicly disclosed).
- A separate 88-compound literature compilation.
Dedupe by InChIKey; expect **~150-250 unique compounds**. Use the established Kp,uu >= 0.1 threshold,
and report the 0.3 threshold as a sensitivity analysis. **State the rat-vs-human caveat prominently**
— most public data is rodent.

**Step 3 — run the head-to-head (1-2 weeks). This is the paper.**
Predict measured Kp,uu class with: your BBB model; CNS-MPO; BBB-Score; LightBBB; ADMETlab;
BBB-Nuke; and your two-axis score. Report AUC/MCC per method on the *exposure* endpoint, next to
each method's published BBB-benchmark AUC. The expected headline is a scatter where benchmark
performance and exposure performance do not line up.

**Step 4 — Leg 1 at the new scale, plus efflux-stratified error analysis** (E1 from
Part 1, requires K-fold retraining for out-of-fold predictions).

**Step 5 — write Paper 1**, then Paper 2 unchanged from Part 1, Paper 2 spec.

---

### 6. Risks, stated honestly

1. **The window is closing.** Five relevant papers in 18 months, the newest in July 2026. If someone
   runs the Kp,uu head-to-head first, this plan is gone. This is the main argument for moving now and
   for preprinting early.
2. **Leg 1 alone is not a paper.** At n = 216 it is a null result on a small sample, and a reviewer
   can call it underpowered or obvious. It only works as the *explanation* for Leg 2. Do not lead
   with it unscaled.
3. **Public Kp,uu data is small and mostly rodent.** 150-250 compounds, rat-dominated. This caps the
   claim's strength and must be disclosed, not buried.
4. **Fridén's 85 compounds are not fresh** — Daood used them. Cite and differentiate (they tested
   predicted substrate status; you test BBB-benchmark-trained models against the exposure endpoint).
5. **Leg 2 might come back positive** — BBB models might predict Kp,uu acceptably. Then the thesis
   inverts and the paper becomes "BBB benchmarks are a better proxy for exposure than their label
   composition suggests," which is publishable but weaker. Write both framings before running it.
6. **Baseline fairness.** If you benchmark BBB-Nuke or LightBBB on Kp,uu compounds that overlap their
   training data, the comparison is void. Check overlap by InChIKey for every baseline and report it.
   BBB-Nuke trains on all of B3DB, so any Kp,uu compound also in B3DB is contaminated for that
   baseline — this will shrink your usable set and you must report the clean subset separately.
7. **A chemistry reviewer will push back on the independence result** by arguing the 216 overlap
   molecules are a biased subset (drugs studied for both endpoints). Pre-empt it: compare their
   property distributions against full B3DB and report the comparison.

---

### 7. If you want the fastest publishable thing instead

If the above is more work than you want, the minimum viable honest paper is **Paper 2 only** from
Part 1, Paper 2 spec: the FDA pharmacopeia two-axis screen plus the 6QEX docking, in
*Molecular Pharmaceutics* or *Pharmaceutics*, framed as a resource and not as a methods advance. It
needs defects A2, A3, B6 fixed and the clinical claims toned down. Novelty is moderate — the
ML-vs-docking-energy correlation in 6QEX is the one piece nobody has published — but it is
achievable in weeks rather than months, and it does not require the new data work.

**My recommendation is still the Kp,uu route.** It is the only path found in 14 searches that leads
to a claim no 2025-2026 paper has made, and it is defendable precisely because the two camps in this
field have not yet been made to talk to each other.

---

### 8. Verification notes

- BBB-Nuke and Daood et al. claims come from **full-text retrieval**, not search snippets. Everything
  else in section 1 comes from search results and should be re-verified against the primary source
  before citation — several are preprints and at least one search result conflated authors
  (ci034205d is Adenot & Lahana, not Muehlbacher).
- Dataset sizes in section 5 are quoted from Daood et al. Table 2 as retrieved; confirm against the
  supporting file once downloaded.
- The 216-molecule statistics were computed in-session from the repo's own files and reconcile
  (45+95+19+57 = 216).
- Yield estimates in Step 1 and Step 2 are **projections, not measurements.** Measure them before
  committing to a framing.


---


# Part 3 — Multi-Task GNN Build Log

**Date:** 2026-10-24
**Repo state at start:** HEAD `e13f3a5 MDs Added`, working tree clean
**Repo state at end:** HEAD unchanged — **nothing committed, nothing pushed**
**Task:** implement the multi-task GNN plan in the repo, make everything run on CUDA,
and log the run.

---

### 1. Scope correction applied mid-run

The plan as approved targeted `Initial Model/`. That was wrong, and was corrected during
the run:

- **`Initial Model/` is the base-paper reference implementation only. It was left
  untouched.** Two files had been created there (`Initial Model/gnn/__init__.py`,
  `Initial Model/gnn/device.py`); both were deleted and `git diff HEAD -- "Initial Model/"`
  confirmed empty. No file in that directory was modified at any point.
- **All work went into `B3DB Based Model/`**, the project's own novel section.

#### Folder rename to "Novel Section": not done

Offered as optional. I kept the name, because `B3DB Based Model` is referenced **74 times
across 6 files**, including `RESEARCH_PAPER_MANUSCRIPT.md`, `README.md`,
`BLOCK_DIAGRAM.md`, Part 1,
`DATASET_AND_FILE_PROVENANCE_CATALOG.md` and `docs/SCIENTIFIC_DOCUMENTATION.md`. Renaming
is a one-command `git mv` plus a find-and-replace that can be done at any time; doing it
now would churn the manuscript and all the architecture docs for no functional gain. The
decision is reversible and costs nothing to defer.

---

### 2. What was built

A new `gnn/` package inside `B3DB Based Model/`, plus three scripts. Nothing pre-existing
was modified.

| File | Lines | Purpose |
|---|---:|---|
| `gnn/__init__.py` | 46 | Package surface |
| `gnn/device.py` | 100 | CUDA selection, GPU tuning, memory accounting |
| `gnn/featurizer.py` | 320 | The single canonical SMILES → graph path |
| `gnn/data_assembly.py` | 256 | Canonicalisation, InChIKey join, corpus merge, masks |
| `gnn/splits.py` | 229 | Molecule-level random + scaffold splits, integrity assertions |
| `gnn/dataset.py` | 164 | Graph construction, disk cache, loaders |
| `gnn/model.py` | 181 | `MultiTaskGNN` — shared trunk, one head per barrier |
| `gnn/losses.py` | 139 | Masked per-task cross-entropy |
| `gnn/metrics.py` | 127 | Metric suite + bootstrap CIs |
| `scripts/train_multitask_gnn.py` | 461 | Training entry point (CUDA) |
| `scripts/verify_multitask_gnn.py` | 294 | 29-check verification suite |
| `scripts/compare_gnn_experiments.py` | 205 | Consolidated comparison table |
| **total** | **2,522** | |

#### Infrastructure that did not previously exist

- **A canonical SMILES / InChIKey join key.** `InChIKey` was generated nowhere in the
  repository, and `core/features.py` has no canonicalisation helper. Without it the two
  benchmarks cannot be merged at molecule level.
- **One shared featurizer.** The reference implementation has two copies that had already
  drifted apart on the valence API (`GetExplicitValence()` vs
  `GetValence(Chem.ValenceType.EXPLICIT)`). `gnn/featurizer.py` has a compatibility shim
  and is the only extractor.
- **Persisted per-molecule predictions.** None existed anywhere in the repo, which blocked
  any error-stratification analysis. Every run now writes a `*_test_predictions.csv`.

---

### 3. CUDA

Everything numeric runs on the GPU. Device selection is centralised in `gnn/device.py`;
no other module decides.

```
device            : cuda:0
GPU               : NVIDIA GeForce RTX 4050 Laptop GPU
memory            : 6.0 GB, compute capability 8.9, 20 SMs
torch             : 2.11.0+cu128   CUDA runtime 12.8   cuDNN 91900
TF32 matmul       : enabled
cudnn.benchmark   : enabled (graph sizes vary, so autotuning pays back over 200 epochs)
pinned memory     : enabled on all loaders when the device is CUDA
host->device      : non_blocking=True
mixed precision   : implemented behind --amp, OFF by default
peak GPU memory   : 0.061-0.065 GB allocated, ~0.10 GB reserved
```

`--amp` is off by default deliberately: these runs take two minutes, so reproducible
numerics are worth more than the speed. `BBB_GNN_FORCE_CPU=1` forces CPU for debugging.

Peak memory is ~1% of the card. The model is small and the bottleneck is Python-side graph
collation, not GPU compute — batch size could go far higher if needed.

---

### 4. Verification: 29/29 checks passed

`python scripts/verify_multitask_gnn.py`

The result that matters most:

> **The legacy featurizer reproduces the base paper's stored reference graphs
> bit-for-bit** — 50 graphs compared, 0 mismatches on `x`, `edge_index` and `edge_attr`.

That keeps the refactor and the feature change separable: any difference in results is
attributable to multi-task training or to the new encoding, not to a quietly different
graph. Legacy mode is confirmed as exactly the base paper's (9, 7) encoding.

Other checks: CUDA reachable and executing; declared feature dims match produced tensors in
both modes; edge lists bidirectional; single-atom molecules and invalid SMILES handled;
corpus label counts reconcile; InChIKeys unique; contingency table sums correctly; both
split protocols pass every integrity assertion with zero scaffold overlap; model forward
contract correct; architecture metadata self-describing.

**Gradient isolation** (the check that makes the masked loss trustworthy):

| Check | Result |
|---|---|
| Batch with no P-gp labels → P-gp head gradient L1 | **0.000e+00** |
| Same batch → BBB head gradient L1 | 3.081e+01 |
| Task weight 0 with 16 P-gp labels present → P-gp head gradient L1 | **0.000e+00** |

---

### 5. Data

| | Rows in | Unique structures | Positive / negative |
|---|---:|---:|---|
| B3DB (`B3DB_classification.tsv`) | 7,807 | 7,805 | 4,956 BBB+ / 2,849 BBB− |
| Broccatelli (`pgp_broccatelli.tab`) | 1,219 | **1,212** | 647 substrate / 565 non-substrate |
| **Merged corpus** | | **8,801** | — |

Reconciles exactly: 7,805 + 1,212 − 216 = 8,801.

- **Dual-labelled: 216** molecules. 97.5% of the corpus carries exactly one label, which
  is why the masked loss is not optional.
- **logBB available for 1,058** molecules — a regression head is possible later at no extra
  data cost.
- **The GNN recovers 7 P-gp molecules the XGBoost arm loses.** That arm's processed file has
  1,205 structures because the ETKDGv3 3D-conformer gate drops molecules that fail to embed.
  Graphs need no conformer, so the GNN trains on 1,212.

#### Label independence on the dual-labelled subset

| | P-gp substrate | P-gp non-substrate |
|---|:---:|:---:|
| **BBB+** | 45 | 95 |
| **BBB−** | 19 | 57 |

χ² = 0.887, p = 0.346; Fisher OR = 1.42, 95% CI [0.758, 2.665]; Cramér's V = **0.064**.
P(P-gp+ | BBB+) = 32.1% vs P(P-gp+ | BBB−) = 25.0%.

Computed independently by `gnn/data_assembly.label_independence()` and matching the earlier
hand analysis to four decimals.

#### Splits (molecule-level, both verified leak-free)

| Protocol | train / val / test | BBB labelled | P-gp labelled | dual-labelled |
|---|---|---|---|---|
| random | 6160 / 1320 / 1321 | 5463 / 1170 / 1172 | 848 / 182 / 182 | 151 / 32 / 33 |
| scaffold | 6160 / 1320 / 1321 | 5637 / 1132 / 1036 | 649 / 230 / 333 | 126 / 42 / 48 |

Scaffold split uses the DeepChem convention (largest clusters fill train, so rare chemotypes
land in test) and reports **zero** scaffold overlap between train and test. This is stricter
than the existing `run_validation.py` protocol, which assigns the largest clusters to test;
**the scaffold numbers below are therefore not comparable to the XGBoost arm's 0.9048.**

#### Feature encodings

| Mode | Atom / bond features | Parameters (1 head) | Parameters (2 heads) |
|---|---|---:|---:|
| legacy (base-paper parity) | 9 / 7 | 345,514 | — |
| encoded (used here) | 66 / 14 | 364,608 | **372,898** |

The second barrier costs **+8,290 parameters, about 2.3%** of the model.

---

### 6. Experiments run (9 total, all on GPU)

| Run | Split | Tasks | Weights | Best epoch | Time |
|---|---|---|---|---:|---:|
| single_bbb | random | bbb | — | 45 | 217s |
| single_pgp | random | pgp | — | 17 | 39s |
| multitask | random | bbb,pgp | 1 / 1 | 35 | 147s |
| single_bbb | scaffold | bbb | — | 7 | 117s |
| single_pgp | scaffold | pgp | — | 19 | 43s |
| multitask | scaffold | bbb,pgp | 1 / 1 | 9 | 98s |
| mt_pgpw3 | random | bbb,pgp | 1 / 3 | 22 | 117s |
| mt_pgpw5 | random | bbb,pgp | 1 / 5 | 18 | 109s |
| sanity_pgpw0 | random | bbb,pgp | 1 / 0 | 63 | 174s |

#### Core result

**RANDOM SPLIT**

| Task | Experiment | n | ROC-AUC | 95% CI | MCC | F1 | BalAcc |
|---|---|---:|---:|---|---:|---:|---:|
| bbb | single_bbb | 1172 | 0.9388 | [0.9246, 0.9531] | 0.7264 | 0.9012 | 0.8613 |
| bbb | multitask | 1172 | 0.9404 | [0.9270, 0.9533] | 0.7078 | 0.8915 | 0.8563 |
| pgp | single_pgp | 182 | 0.9390 | [0.8964, 0.9728] | 0.7460 | 0.8821 | 0.8727 |
| pgp | multitask | 182 | 0.9301 | [0.8810, 0.9688] | 0.7460 | 0.8821 | 0.8727 |

**SCAFFOLD SPLIT**

| Task | Experiment | n | ROC-AUC | 95% CI | MCC | F1 | BalAcc |
|---|---|---:|---:|---|---:|---:|---:|
| bbb | single_bbb | 1036 | 0.7583 | [0.7288, 0.7862] | 0.3927 | 0.7677 | 0.6783 |
| bbb | multitask | 1036 | 0.7467 | [0.7180, 0.7753] | 0.2879 | 0.7461 | 0.6130 |
| pgp | single_pgp | 333 | 0.8227 | [0.7763, 0.8696] | 0.5442 | 0.7990 | 0.7605 |
| pgp | multitask | 333 | 0.8171 | [0.7720, 0.8616] | 0.5389 | 0.7902 | 0.7660 |

**Multi-task effect (multi minus single):**

| Task | Split | ΔROC-AUC | ΔMCC | ΔF1 | Verdict |
|---|---|---:|---:|---:|---|
| bbb | random | +0.0016 | −0.0186 | −0.0097 | within noise (CIs overlap) |
| bbb | scaffold | −0.0116 | −0.1048 | −0.0216 | within noise (CIs overlap) |
| pgp | random | −0.0089 | +0.0000 | +0.0000 | within noise (CIs overlap) |
| pgp | scaffold | −0.0056 | −0.0053 | −0.0088 | within noise (CIs overlap) |

---

### 7. The finding: the pre-registered hypothesis is not supported

The plan pre-registered an asymmetric expectation — the small P-gp task should gain from
sharing an encoder with the large BBB task, while BBB might not gain. **Neither half holds.
Multi-task training is neutral to mildly harmful on both tasks, on both splits.**

This is a real negative result, not a bug, and three pieces of evidence establish that:

1. **The plumbing is verified.** `sanity_pgpw0` (multi-task architecture, P-gp weight 0)
   gives BBB ROC-AUC **0.9411** against the single-task baseline's 0.9388 — within noise —
   while its untrained P-gp head sits at **0.5626**, i.e. chance. Weight 0 genuinely removes
   a task, and the trunk trains exactly as it does single-task.
2. **The obvious confound is ruled out.** P-gp has 848 training molecules against BBB's
   5,463, so thin gradient signal was the likeliest innocent explanation. Upweighting P-gp
   does **not** rescue it (ROC-AUC 0.9295 at weight 3, 0.9299 at weight 5, versus 0.9390
   single-task) and progressively **damages BBB** (0.9388 → 0.9310 → 0.9086, MCC
   0.7264 → 0.6821 → 0.6467). The tasks compete for encoder capacity instead of helping
   each other.
3. **It coheres with the label measurement.** Cramér's V = 0.064 means the two labels share
   almost no information. Multi-task learning borrows strength from *correlated* auxiliary
   signal; near-independent labels have none to lend. The negative result is what the
   independence measurement predicts.

A 2025 review proposed exactly this architecture as future work — "multi-task learning
leverages auxiliary signals such as transporter affinity to guide the extraction of
disentangled, mechanism-relevant features." This run tests that proposal on the two standard
public benchmarks and finds it does not deliver, with a mechanistic explanation for why.

#### Chemotype transfer

| Experiment | Task | random → scaffold ROC-AUC |
|---|---|---|
| single_bbb | bbb | 0.9388 → 0.7583 (**−0.1805**) |
| multitask | bbb | 0.9404 → 0.7467 (−0.1937) |
| single_pgp | pgp | 0.9390 → 0.8227 (**−0.1163**) |
| multitask | pgp | 0.9301 → 0.8171 (−0.1130) |

BBB degrades ~55% more than P-gp under chemotype shift (−0.18 vs −0.12), reproducing in a
GNN the asymmetry previously measured in the XGBoost arm. The permeability axis is the
chemotype-dependent one; the efflux axis transfers better.

---

### 8. Two bugs found and fixed during the run

**PyG silently corrupted the row pointer.** Graphs carried a `corpus_index` attribute to
trace predictions back to the corpus. PyG's default `Data.__inc__` increments *any*
attribute whose key contains `index` by the graph's node count when collating a batch — so
the pointer came out offset by cumulative node counts and `corpus.iloc[...]` raised
"positional indexers are out-of-bounds" only at the prediction-saving step, after training
had completed. Renamed to `corpus_row`, with a cache version bump so stale caches rebuild
rather than being silently reused. Had this gone unnoticed in a form that did not crash,
every saved prediction would have been attributed to the wrong molecule.

**`torch.cuda.reset_peak_memory_stats(int)` raised "Invalid device argument"** when called
before the CUDA context was initialised. Fixed by passing the `torch.device` object and
calling `torch.cuda.init()` first.

---

### 9. Pre-existing defects confirmed but NOT fixed

These are in `Initial Model/` and were left alone per the scope correction. They are
recorded because they affect any number quoted from that arm:

- `train_gnn.py` discards 269 Li molecules (14.2% of 1,890), so all its metrics are
  Adenot-only, and dropping Li makes imbalance *worse* (79.1% positive vs Li's 62.8%).
- `predict.py` cannot load its own default checkpoint: it hardcodes `HIDDEN_CHANNELS = 256`
  and ignores `checkpoint["hidden_channels"]`, while `best_bbb_gnn_model.pt` is a 128-dim
  artifact → `load_state_dict` size mismatch.
- `predict.py` also re-declares the model without dropout and substitutes
  `torch.mean(x, dim=0, keepdim=True)` for `global_mean_pool`.
- No metrics are persisted anywhere in that arm; `confusion_matrix` is imported and never
  called.

The new `gnn/` package avoids all four by construction: one shared featurizer, checkpoints
that describe their own architecture, and every run writing metrics to disk.

**The defects listed in Part 1, Defect Register for the XGBoost arm (unscaled
SHAP, hard-coded docking probabilities, the two legacy evaluation scripts) are also still
unfixed.** This run did not touch them.

---

### 10. Artifacts produced

- **12 new source files** under `B3DB Based Model/gnn/` and `B3DB Based Model/scripts/`.
- **39 result files** in `B3DB Based Model/results/` — per run: a report JSON (full
  provenance: arguments, device, corpus stats, independence statistics, split report,
  architecture, metrics), a metrics CSV, an epoch-by-epoch history CSV, and a per-molecule
  test-prediction CSV. Plus `gnn_experiment_comparison.csv` and `gnn_multitask_delta.csv`.
- **9 checkpoints** in `B3DB Based Model/models/`, each ~1.5 MB and self-describing.
- **1 graph cache** at `B3DB Based Model/data/graph_cache/` (122 MB).

#### One repo-hygiene decision left to you

`.gitignore:10` is `*.pt`, so **all 9 new GNN checkpoints and the graph cache are currently
ignored** — the same reproducibility gap the `Initial Model/` arm has. The 122 MB cache
should stay ignored. The checkpoints total ~13.5 MB and arguably should not. If you want
them tracked, add after the `*.pt` line:

```
!B3DB Based Model/models/gnn_*.pt
```

I have not changed `.gitignore` — what belongs in version control is your call.

---

### 11. What this means for the paper

The plan's decision gate fires: *"if the P-gp head does not improve, the headline shifts to
C2+C3+C4 — the measurement, the evaluation, and the attribution — with multi-task reported
as a negative result."*

So the paper is no longer "we built the first joint BBB/efflux GNN and it works." It is:

> **Blood-brain barrier and P-glycoprotein efflux labels are near-independent, and
> multi-task graph learning cannot bridge them.** We quantify the label independence on the
> 216 molecules the two standard benchmarks share, show that a shared-encoder multi-task GNN
> delivers no benefit to either task under either split protocol, demonstrate that
> upweighting the minority task causes measurable negative transfer rather than gain, and
> show the permeability axis degrades ~55% more than the efflux axis under chemotype shift.

That is a defendable contribution: it tests a published proposal, reports a clean null with
the confounds eliminated, and explains the null mechanistically. Negative results on a
proposal that a review named as promising are publishable in *Journal of Cheminformatics*,
and the released cross-labelled 216-molecule resource is the citable artifact.

**Still open and worth doing**, in order: the Kp,uu external validation (the strongest
remaining novelty per Part 2 — no competitor validates against measured
brain exposure); dual-head atom attribution to test graphB3's untested P-gp hypothesis; the
encoded-vs-legacy feature ablation (one flag: `--features legacy`); and a logBB regression
head using the 1,058 available values.

**One caveat to carry forward:** every number here is a single seed. Before any of it goes
into a manuscript, repeat across seeds — the multi-task deltas are small enough that
seed variance could account for them, which is itself part of the negative-result argument
but needs to be measured rather than assumed.

---

### 12. Reproducing this run

```bash
cd "B3DB Based Model"
python scripts/verify_multitask_gnn.py                                   # 29 checks
python scripts/train_multitask_gnn.py --tasks bbb      --split random    --tag single_bbb
python scripts/train_multitask_gnn.py --tasks pgp      --split random    --tag single_pgp
python scripts/train_multitask_gnn.py --tasks bbb,pgp  --split random    --tag multitask
python scripts/train_multitask_gnn.py --tasks bbb      --split scaffold  --tag single_bbb
python scripts/train_multitask_gnn.py --tasks pgp      --split scaffold  --tag single_pgp
python scripts/train_multitask_gnn.py --tasks bbb,pgp  --split scaffold  --tag multitask
python scripts/train_multitask_gnn.py --tasks bbb,pgp  --split random --task-weights pgp=3 --tag mt_pgpw3
python scripts/train_multitask_gnn.py --tasks bbb,pgp  --split random --task-weights pgp=5 --tag mt_pgpw5
python scripts/train_multitask_gnn.py --tasks bbb,pgp  --split random --task-weights pgp=0 --tag sanity_pgpw0
python scripts/compare_gnn_experiments.py
```

Total wall-clock for all nine runs plus verification: about 19 minutes on an RTX 4050.

---
---

## Session 2: Decision-Aware Metrics, K-Fold, and the Statistical Rigor Layer

**Date:** 2026-10-24 (same day, second work block)
**Repo state at start:** HEAD `e13f3a5`, Session 1 work uncommitted
**Repo state at end:** HEAD unchanged — **nothing committed, nothing pushed**
**Task:** add a novel metric contribution, plus the K-fold and seed-variance work needed
to make Session 1's null result defensible.

### 1. Why this session happened

Session 1 left two problems. The multi-task null rested on **one seed**, so deltas of
+0.0016 to −0.0116 could not be told from noise. And the project has no performance
story — its BBB ROC-AUC of 0.9388 sits below graphB3's 0.943 — so if there is a
contribution it has to be in *measurement*.

Four literature searches established what was unclaimed:

- **No paper proposes joint or quadrant metrics for multi-endpoint ADMET.** The field
  states the need itself ("ROC-AUC is a global metric that is not always useful for
  making decisions in a production setting"); TDC still uses per-task metrics only.
- **No prior art for efflux-stratified performance reporting.**
- **Nothing frames model selection as expected regret over a screening decision.** The
  closest is a 2026 preprint (Budget-Sensitive Discovery Scoring) whose cost-ratio
  parameterisation λ = c_FP/v_hit is adapted here and cited.
- **MDE-over-seeds reporting is NOT novel.** A 2026 PROTAC-activity preprint does exactly
  this, reporting a +0.0025 AUROC effect as "informative as a null result". It is adopted
  as a rigor layer and cited, never claimed.

### 2. What was built

| File | Lines | Purpose |
|---|---:|---|
| `gnn/training.py` | 283 | Shared training loop, extracted so three runners share one implementation |
| `gnn/decision_metrics.py` | 341 | M1–M5, the metric suite |
| `scripts/run_kfold_multitask.py` | 329 | K-fold runner producing out-of-fold predictions |
| `scripts/run_metric_analysis.py` | 248 | Scores the three decision strategies on M1–M5 |
| `scripts/run_seed_variance.py` | 300 | Multi-seed runs, MDE, P(A>B) |
| `scripts/verify_decision_metrics.py` | 262 | 31-check metric verification suite |
| `core/decision_matrix.py` | +33 | Added `quadrant_id`, `QUADRANT_NAMES`, `QUADRANT_IDS` |

**One refactor of existing code.** `scripts/train_multitask_gnn.py` went from 461 to 353
lines: its local copies of `set_all_seeds`, `parse_task_weights`, `move_batch`, `run_epoch`
and `quick_f1` were deleted and are now imported from `gnn/training.py`. Three copies of an
early-stopping loop would have drifted apart exactly the way the base paper's two
featurizers did.

**Quadrant convention is single-sourced.** `quadrant_id` was added to the *existing*
`core/decision_matrix.py` rather than reimplemented in the metrics module, and is verified
to agree with `determine_dual_barrier_quadrant` on all four inputs and to reproduce
`expected_quadrant` for all five `BENCHMARK_DRUGS`.

### 3. The metric suite

| | Metric | Definition |
|---|---|---|
| M1 | Quadrant exact-match | Both axes correct (subset accuracy) |
| M2 | Quadrant κ + 4×4 matrix | Chance-corrected 4-class agreement, with the error *kind* visible |
| M3 | Paradox recall / leakage | Recall on Q2, and the share of true Q2 molecules progressed as Q1 |
| M4 | Efflux-disparity gap | AUC_BBB(non-substrates) − AUC_BBB(substrates) |
| M5 | Decision regret | Expected cost of the progression decision, as a curve in λ |

The mathematics is standard. The contribution is the framing: these are the numbers that
decide a CNS progression call, and none of them is currently reported.

### 4. Verification: 31/31 metric checks, 29/29 regression

`python scripts/verify_decision_metrics.py` → **31/31**. The checks that matter:

| Check | Result |
|---|---|
| Perfect predictor | exact-match 1.0, κ 1.0, paradox recall 1.0, leakage 0.0, zero regret at every λ |
| Single-axis BBB model | paradox recall exactly **0**, leakage exactly **1.0** — by construction |
| All-Q2 predictor | recall 1.0, precision equals Q2 prevalence (0.2625 vs 0.2625) |
| Confusion matrix | sums to n; diagonal equals the exact-match count |
| EDG, equally informative strata | gap 0.0, CI spans zero |
| EDG, signal destroyed on substrates only | gap +0.5305, CI [0.4668, 0.5914] excludes zero |
| EDG, single-class stratum | returns a note instead of a number |
| Regret at λ=1 | equals the Q1-vs-rest error rate exactly |
| Regret at λ=10⁶ | dominated by false progressions (residual 1.1e-6) |
| MDE formula | matches hand calculation to 6 dp (0.016625) |
| P(A>B) | matches hand values exactly (0.7778, 0.5000, 1.0000) |

**Regressions after the refactor:** `verify_multitask_gnn.py` still **29/29**, and
`compare_gnn_experiments.py` reproduces the Session 1 table byte-identically.

**Quadrant ground truth reconciles:** Q1 = 95, Q2 = 45, Q3 = 57, Q4 = 19, sum **216**.

### 5. K-fold out-of-fold predictions

Session 1's single-split test partitions held only 33 (random) / 48 (scaffold) dual-labelled
molecules — too few to stratify. 5-fold stratified K-fold over all 8,801 molecules gives
out-of-fold predictions for **216 of 216**.

| Run | Folds | Integrity | Pooled OOF result | Time |
|---|---|---|---|---|
| single_bbb | 5 | 7,805 scored once, no leaks | BBB ROC-AUC **0.9379** [0.9323, 0.9433] | 771s |
| single_pgp | 5 | 1,212 scored once, no leaks | P-gp ROC-AUC **0.9011** [0.8838, 0.9190] | 142s |
| multitask | 5 | 8,801 scored once, no leaks, **216/216 dual** | BBB **0.9363** [0.9312, 0.9416], P-gp **0.9056** [0.8885, 0.9226] | 934s |

Every fold's assertions passed: no molecule scored more than once, every labelled molecule
covered, and no molecule scored by a fold that trained on it.

On the much larger pooled OOF sample the per-axis picture is the same as Session 1: BBB
−0.0016 and P-gp +0.0045 for multi-task, both with heavily overlapping intervals.

### 6. Decision-metric results

Three strategies scored on the 216 dual-labelled molecules:

- **A. BBB-only** — single-axis model; scoring it means assuming P-gp-negative throughout,
  which is what a BBB-only pipeline implies. This is the published status quo.
- **B. Composed two-model** — independent single-task BBB and P-gp models combined. Valid
  because both K-fold runs share an identical fold assignment (verified at runtime).
- **C. Multi-task** — one shared encoder, two heads.

#### M1–M3

| Evaluation | Exact-match | κ | Paradox recall | Paradox precision | Leakage |
|---|---:|---:|---:|---:|---:|
| A BBB-only | 0.5556 [0.4769, 0.6204] | 0.2742 | **0.0000** | 0.0000 | **0.9333** |
| B Composed | **0.5787** [0.5093, 0.6481] | **0.3725** | 0.4444 [0.2954, 0.6047] | 0.4348 | 0.4889 |
| C Multi-task | 0.5602 [0.4954, 0.6251] | 0.3478 | **0.5111** [0.3657, 0.6667] | **0.4792** | **0.4222** |

**The headline: per-axis ROC-AUC is ~0.94, yet only 56–58% of molecules land in the correct
quadrant.** Four in ten joint decisions are wrong. No currently reported metric shows this.

**Paradox leakage** — true Q2 (permeant but effluxed) molecules progressed as Q1:

| | Leaked | Rate |
|---|---|---|
| A BBB-only | **42 of 45** | **93.3%** |
| B Composed | 22 of 45 | 48.9% |
| C Multi-task | 19 of 45 | 42.2% |

A single-axis model advances 93% of paradoxical compounds as though they reach the brain.
A second axis roughly halves that. B and C are indistinguishable from each other.

#### M4 — the hypothesis was wrong, and the mechanism explains why

I predicted BBB prediction would be *worse* on efflux substrates. It is significantly
**better**:

| Evaluation | AUC non-substrate (n=152) | AUC substrate (n=64) | Gap | Verdict |
|---|---:|---:|---:|---|
| A / B | 0.7990 | 0.9368 | **−0.1379** [−0.2406, −0.0410] | gap detected |
| C | 0.7904 | 0.8760 | −0.0856 [−0.1992, +0.0247] | CI spans zero |

Why, measured rather than assumed. The two strata differ sharply in chemistry:

| | MolWt | LogP | TPSA | HDonors | RotBonds |
|---|---:|---:|---:|---:|---:|
| non-substrate | 281.4 | 2.2 | 60.3 | 1.4 | 3.9 |
| substrate | 370.9 | 3.8 | 47.8 | 0.8 | 5.2 |

And BBB permeability is *more* separable among substrates: the BBB+ vs BBB− TPSA gap is
**−28.9** there versus **−21.5** among non-substrates. P-gp substrates are a chemically
narrow, drug-like region where polarity cleanly determines permeability; non-substrates are
a heterogeneous mix including small polar molecules where permeability is genuinely harder
to call.

**This sharpens the paper rather than weakening it.** The harm from unmodelled efflux is not
that permeability prediction degrades — it is that permeability prediction is *accurate*, and
that accurate prediction is clinically misleading. The molecule really is permeant and really
is pumped back out. The 93.3% leakage is the harm; there is no accuracy deficit to point at.

#### M5 — the method ranking depends on the cost ratio

| Evaluation | λ=0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A BBB-only | **0.1296** | **0.2083** | 0.3657 | 0.6806 | 1.3102 | 2.5694 | 5.0880 |
| B Composed | 0.1806 | 0.2268 | **0.3194** | **0.5046** | **0.8750** | **1.6157** | **3.0972** |
| C Multi-task | 0.1921 | 0.2407 | 0.3380 | 0.5324 | 0.9213 | 1.6991 | 3.2546 |

False progressions by true quadrant: A {Q2:42, Q3:21, Q4:5}, B {Q2:22, Q3:16, Q4:2},
C {Q2:19, Q3:19, Q4:4}.

Regret is linear in λ, so the crossover solves exactly: **λ = 0.643** against B and
**λ = 0.769** against C. **Below that, the single-axis model is genuinely the better
decision rule** — it progresses more compounds and incurs fewer missed opportunities
(11 vs 29 and 31). The efflux axis only pays once a false progression costs more than
roughly 0.7× a missed opportunity.

That nuance is the argument for reporting a curve. A single headline number would have
hidden it, and the crossover calculation is now part of the script rather than a one-off.

### 7. Seed variance and minimum detectable effect

5 seeds, paired by seed (the same seed drives the split *and* the initialisation for every
experiment), random split.

Per-seed ROC-AUC:

| Seed | BBB multi | BBB single | P-gp multi | P-gp single |
|---|---:|---:|---:|---:|
| 42 | 0.9443 | 0.9462 | 0.9464 | 0.9445 |
| 43 | 0.9331 | 0.9428 | 0.8734 | 0.8734 |
| 44 | 0.9340 | 0.9295 | 0.9195 | 0.9122 |
| 45 | 0.9163 | 0.9396 | 0.8841 | 0.9300 |
| 46 | 0.9286 | 0.9456 | 0.8933 | 0.8701 |

| | BBB | P-gp |
|---|---|---|
| multi-task mean | 0.9313 | 0.9033 |
| single-task mean | 0.9407 | 0.9060 |
| delta | **−0.00944** | **−0.00272** |
| delta 95% CI | [−0.02337, +0.00449] | [−0.03479, +0.02936] |
| sd of differences | 0.01122 | 0.02583 |
| **MDE at 5 seeds (80% power)** | **0.01865** | **0.04295** |
| abs(delta) below MDE | **yes** | **yes** |
| paired t-test | t = −1.881, p = 0.1331 | t = −0.235, p = 0.8256 |
| **P(multi > single)** | **0.200** | 0.540 |
| seeds needed for 80% power at this effect | **14** | **712** |

**What this licenses, stated precisely.** Neither delta is resolvable at 5 seeds, so the
correct conclusion is *"no effect detected at this sample size"* — never *"no effect
exists."* But the two tasks are not in the same position:

- **BBB:** P(multi > single) = **0.200**, meaning single-task wins four cross-seed pairings
  in five. By Bouthillier et al.'s own 0.75/0.25 criterion that is a reliable *ordering*,
  even though the paired t-test is non-significant at p = 0.133. The evidence leans toward
  multi-task being mildly harmful to BBB, and **14 seeds would settle it.**
- **P-gp:** P(multi > single) = **0.540**, indistinguishable from a coin flip, and resolving
  the −0.0027 delta would need **712 seeds**. An effect that small is practically zero
  whatever its sign. This is the strongest form the P-gp null can take.

A footnote worth recording: at seed 43 the two P-gp models produced identical ROC-AUC to
four decimals (0.8734). With n = 182 the AUC granularity is 1/8245, so a collision is
possible, and the other four seeds differ — there is no systematic identity. Not chased
further.

### 8. Two problems found and fixed

**Stored metrics were rounded to 4 decimals**, which makes exact delta arithmetic impossible
when the effects under study are of order 1e-3. Caught by a verification check that compared
the confusion-matrix diagonal against the exact-match rate and failed at 1e-9 tolerance. The
correct fix was to store at 6 dp and format to 4 only for display, not to loosen the test.

**The first background launch used `nohup` plus `&`**, which detached the child from the
tool's process management; it was killed immediately and reported success with an empty log.
Re-running through the tool's own backgrounding worked. A job that "succeeds" with no output
is worth checking rather than trusting.

### 9. Artifacts added this session

- **7 source files** (6 new, 1 modified) under `B3DB Based Model/gnn/` and `scripts/`.
- **3 OOF prediction files** covering every labelled molecule, with both axes for all 216
  dual-labelled ones.
- **3 K-fold report JSONs** with full provenance.
- **`results/gnn_decision_metrics.csv`** and **`_report.json`** — the M1–M5 tables.
- **4 seed-variance files** — per-run, summary, contrasts, report.

### 10. What this means for the paper

The claim is now three-part and each part is backed:

1. **BBB and P-gp labels are near-independent** (Cramér's V = 0.064, n = 216) — recomputed
   independently this session and matching to four decimals.
2. **Multi-task graph learning does not bridge them.** No effect detected on either task,
   on either split, on single-split or pooled K-fold; upweighting the minority task causes
   measurable negative transfer; and the plumbing is verified. For BBB the ordering favours
   single-task at P = 0.20 and 14 seeds would confirm it; for P-gp the effect is below any
   feasible detection threshold.
3. **Per-axis metrics hide the decision that matters.** ROC-AUC ~0.94 on both axes, yet
   56–58% quadrant accuracy and 93.3% paradox leakage for the single-axis status quo.

Contribution 3 is the new one and it is a *supporting* contribution: the mathematics is
standard, so the defence is that nobody evaluates this decision jointly. A determined
reviewer may still call it incremental, and it does not substitute for the Kp,uu validation
that Part 2 identifies as the strongest remaining novelty.

**Still open**, in order: Kp,uu external validation; dual-head atom attribution to test
graphB3's untested P-gp hypothesis; the `--features legacy` encoding ablation; a logBB
regression head on the 1,058 available values; and 14 seeds on the BBB contrast to convert
"not detected" into "ordered."

**The scaffold-split decision metrics were not computed.** Only the random-split K-fold was
run. Repeating the K-fold under the scaffold protocol would show whether the quadrant and
leakage numbers survive chemotype shift, and should be done before any of these figures go
into a manuscript.

### 11. Reproducing this session

```bash
cd "B3DB Based Model"
python scripts/verify_decision_metrics.py                                    # 31 checks
python scripts/verify_multitask_gnn.py                                       # 29 checks
python scripts/run_kfold_multitask.py --tasks bbb     --folds 5 --tag single_bbb
python scripts/run_kfold_multitask.py --tasks pgp     --folds 5 --tag single_pgp
python scripts/run_kfold_multitask.py --tasks bbb,pgp --folds 5 --tag multitask
python scripts/run_metric_analysis.py
python scripts/run_seed_variance.py --seeds 42,43,44,45,46 --split random
```

Wall-clock on the RTX 4050: K-fold 31 min, seed variance 18 min, analysis seconds.


---

# Part 4 — Prior Audit Disposition

An independent code audit was produced on 2026-10-19 as `B3DB_Model_Analysis.pdf`
and raised eight findings. Each was re-verified against the current code on
2026-10-24; all are resolved. The PDF has been removed and its findings recorded
here, because an audit document that is entirely actioned is history, not guidance.

| # | Finding | Status | Evidence |
|---|---|---|---|
| 1 | P-gp fingerprints computed on the H-added molecule in training but not at inference (~57 bits different) | **Resolved** | Both paths now use the non-H molecule. Verified empirically: 200/200 stored rows match the non-H fingerprint exactly; the H-added version differs by 57.7 bits on average, matching the audit's figure. |
| 2 | Optuna scored trials on the test set, so 0.954 AUC was optimistic | **Resolved** | `bbb_model_training.py` now uses a three-way split and `objective()` receives `X_val`, not `X_test`. |
| 3 | Scalers fit on all data before splitting | **Resolved** | `scaler.fit_transform(X_train[...])` runs after the split; val and test use `.transform()`. |
| 4 | "Scaffold-aware" evaluation was only a scaffold count, not a split | **Resolved** | `run_validation.py` implements `generate_bemis_murcko_split` and reports a genuine zero-overlap scaffold split. |
| 5 | Manuscript ablation standard deviations about half those in the CSV | **Resolved** | All 24 cells of the ablation table now match `ablation_study_results.csv` exactly. |
| 6 | README oversold "3D conformation-augmented" as a key capability | **Resolved** | The claim is gone. The 3D result is now reported as the finding it is: shape descriptors alone reach 0.8425 ROC-AUC but add nothing once Morgan fingerprints are present. |
| 7 | Benchmark drugs might be in the training sets (unverified at the time) | **Resolved, and the audit was right** | Verified by InChIKey: all five are in B3DB and three are also in Broccatelli. The manuscript's "100% ABSENT" blind-validation claim was withdrawn. The claim had survived only because the Loperamide SMILES was the wrong molecule. |
| 8 | Smaller items: hand-written scaler metadata, deprecated RDKit call, `os.system('color')`, AD sampling, low counterfactual similarity | **Mostly resolved** | Scaler metadata is written by the training script; fingerprints use `rdFingerprintGenerator`. Two cosmetic items remain: one deprecated `GetMorganFingerprintAsBitVect` call, and a Windows-only `os.system('color')` in `inference.py`. |

Finding 7 is the one worth remembering. The audit flagged it as *unverified*, and it
turned out to be true — but it could not be confirmed at the time because the
structure being checked was not the drug it was labelled as. A data-integrity
problem was masking a methodology problem.
