# Novelty Assessment: What Is Publishable In This Repository

**Prepared:** 2026-10-24
**Scope:** audit of `RESEARCH_PAPER_MANUSCRIPT.md` against current literature, plus a measured
result that changes what this project can claim.
**Status:** analysis only. No code was modified, no experiment was run, nothing was committed.

---

## 1. Verdict

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

## 2. What will not survive review

| Manuscript claim | Status | Prior art / evidence |
|---|---|---|
| "Unified dual-barrier framework" (contributions 1, 5) | **Scooped, 2004** | Adenot & Lahana, *J. Chem. Inf. Comput. Sci.* **2004**, 44, 239-248 built a BBB model *and* a P-gp substrate model and proposed a combined CNS map explicitly including P-gp substrates. **The PDF is in this repo** at `Initial Model/reference_papers/adenot_2004_jcim.pdf`, and the paper is **absent from the reference list.** |
| "via 3D Conformational Modeling" (title, contribution 2) | **Scooped *and* self-refuted** | Tiwari et al., *Pharmaceutics* **2026**, 18, 1018 reached the same negative conclusion on B3DB. Worse, `results/ablation_study_results.csv` shows Morgan+2D at ROC-AUC **0.9509** beating the deployed Morgan+2D+3D at **0.9481**, and 2D-only 0.9185 vs 2D+3D 0.9192 (+0.0007). The title claims a benefit the project's own table refutes. |
| TreeSHAP + Morgan bit deconvolution; TPSA as top BBB feature (contributions 3, 4) | **Scooped** | Fingerprint-based explainable BBB ML, bioRxiv **2025**.10.25.684495. Tiwari et al. 2026 also report TopoPSA as the top SHAP feature, tied to desolvation: the same finding and the same interpretation. |
| Scaffold-split robustness (section 4.1) | **Scooped, and spun backwards** | *J. Chem. Inf. Model.* **2026**, 66, 6837 ("Overrepresentation Bias..."); Guo et al. **2024**, "Scaffold Splits Overestimate Virtual Screening Performance". Section 4.1 says the split "prov[es] substantial generalizability" while its own Table 1 shows specificity **0.848 to 0.580** and MCC **0.754 to 0.572**. Reviewers read the table. |
| ROC-AUC 0.9522 on B3DB | **Mid-pack** | Published B3DB results cluster at 0.93-0.95; a 2025 geometric GNN reports 0.9685 *under scaffold splits* (arXiv 2507.18926). This cannot be a performance paper. |
| Applicability domain, SMARTS counterfactuals, quadrant matrix (contribution 5) | **Standard toolkit** | Competent engineering, not contributions. |

### What *is* defensible

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

## 3. The novel result

### 3.1 The overlap set

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

### 3.2 The claim, stated at two strengths

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

### 3.3 Supporting finding: the two axes generalize differently

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

### 3.4 A second cross-labelled source (weak, caveated)

The Adenot sheet of `Initial Model/Dataset/raw_adenot_li.xls` has a `P-gp` column. It is **not a
label**: it holds provenance strings ("P-gp (Adenot, J. Chem. Inf. Comput. Sci. 2004, 44,
239-248)") marking 88 of 1,632 rows as drawn from P-gp substrate compilations. Usable only as an
88-molecule supplementary check, and only if the semantics are stated precisely.

Worth noting in passing: `Initial Model/clean_raw_data.py` selects
`["SMILES", "SMILES-B", "Name", "BBB-crossing"]` and discards this column, so the pipeline throws
away its only other cross-labelled annotation.

---

## 4. Paper 1: methods

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

### E1: efflux-stratified error analysis (specification; not run)

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

## 5. Paper 2: translational

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

### E2: re-run the docking correlation with live model output (specification; not run)

`scripts/run_cryoem_docking_validation.py` hard-codes `prob_pgp_ml` as 22 literal constants (first
at line 66). So section 6.4's "cross-modality correlation" is docking energies against numbers typed
into a script. The arithmetic *is* reproducible from the saved CSV (I recomputed rho = -0.5965,
p = 0.004315), but nothing in the pipeline ties that column to the model, and a reviewer who opens
the script will see it. Replace the literals with a live call to the saved booster through
`core/features.prepare_dataframe` plus `pgp_scaler_metadata.json`.

### Honesty requirements for Paper 2

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

## 6. Defect register

### A. Invalidating: these results are currently wrong

**A1. Global SHAP is computed on unscaled inputs.**
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

**A2. Docking ML probabilities are hard-coded literals.** See E2 above.

**A3. Two legacy evaluation scripts feed unscaled features to scaled boosters.**
`BBB/bbb_model_evaluation.py` and `Pgp/pgp_model_evaluation.py` both pass raw `X` to a model trained
on z-scored input, and `bbb_model_evaluation.py` additionally uses an 80/20 *unstratified* split.
Their metrics are invalid and two figures derived from them are still tracked. Delete both, or fix
and regenerate.

### B. Internal contradictions a reviewer will catch

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

### C. Citation and data errors

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

### D. Statistical reporting

**D10. A non-significant correlation is presented as a finding.** Section 4.3 reports rho = 0.7714
as the "Cross-Task Correlation" supporting the shared-property argument, but **p = 0.0724** at
n = 6. State it as non-significant, or drop the claim. (The value itself reproduces exactly from the
current CSV.)

**D11. Checked and fine, no action.** Section 6.2's Mann-Whitney p = 0.0043 matching the Spearman p
is a rounding coincidence, not a copy error: recomputed, MWU = 0.004274 and Spearman = 0.004315, and
both round to 0.0043.

### E. Framing

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

### Out of scope for both papers

The GNN in `Initial Model/` should appear in neither manuscript. `train_gnn.py` contains
`if graph.get("source", "") == "Li": continue`, silently discarding all 269 Li molecules from a
1,890-molecule corpus (14.2%), so every reported metric is Adenot-only on about 146 test molecules
per fold. And **no GNN metrics are persisted anywhere** (12 `.pt` checkpoints exist, zero result
files), so any number quoted for it would have to come from prose or a re-run.

---

## 7. Asset division

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

## 8. Sequencing

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

## 9. Sources consulted

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
