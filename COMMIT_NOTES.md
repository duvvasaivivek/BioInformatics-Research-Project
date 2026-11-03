# Commit Notes: The Pharmacopeia Screen, the Cryo-EM Validation, and the README Rewrite

## Scope

This document explains three commits from **Tuesday 20 October 2026**, which landed within 42 minutes of each other and together form a single piece of work:

| # | SHA | Subject |
|---|-----|---------|
| 1 | `53a1577` | `feat(repurposing)` — FDA pharmacopeia virtual screen: 273 repurposing leads, 105 efflux resistance profiles |
| 2 | `7378e3a` | `feat(docking)` — Cryo-EM human P-gp (PDB 6QEX) docking validation across 22 drugs |
| 3 | `325ef77` | `docs(readme)` — chronological scientific workflow, mermaid pipeline diagrams |

The arc is **discover → validate → document**: commit 1 produces a computational claim, commit 2 tests that claim against an independent biophysical modality, commit 3 makes both legible to a reader arriving cold.

> **Path warning.** All three commits were authored under a directory named `B3DB Based Model/`. That directory **no longer exists** — commit `a53789f` renamed it to `Novelty Upgrade/`. Every path quoted from these commits must be re-rooted before you try it at `HEAD`. Likewise, the two XGBoost model files were later renamed (`bbb_xgboost_optuna_best.json` → `bbb_xgboost.json`, `pgp_xgboost_model.json` → `pgp_xgboost.json`).

---

## 1. Where these commits sit in the history

```
eb3b3ea  Merge PR #2 - data-leakage fixes, Bemis-Murcko scaffold split      <- the ground these commits stand on
53a1577  feat(repurposing): FDA pharmacopeia virtual screen                 <- COMMIT 1
7378e3a  feat(docking): Cryo-EM 6QEX docking validation                     <- COMMIT 2
325ef77  docs(readme): chronological workflow + mermaid diagrams            <- COMMIT 3
a53789f  Restructured Codebase            (B3DB Based Model/ -> Novelty Upgrade/)
5518b82  Major Fixes
760d77a  Remove dead code, delete defective evaluation scripts
dfbbc5b  Fix unscaled SHAP, hard-coded docking probabilities, bad Topotecan SMILES
afe832e  Audit the five clinical benchmark SMILES (Loperamide, Penicillin G)
...
b44c754  Consolidate documentation to two public documents                   <- replaced commit 3's README wholesale
79d4157  (HEAD) Restore gnn_decision_metrics.csv
```

Two consequences follow, and both are stated plainly in this document rather than buried:

- The **code** from commits 1 and 2 survives at `HEAD`, but with defects corrected (§6).
- The **README** from commit 3 does **not** survive. `b44c754` rewrote it from scratch into the current two-public-document structure. Commit 3 is therefore of historical interest: it explains how the project chose to present itself at that moment, not how it presents itself now.

---

## 2. Commit 1 — `53a1577`: Virtual screening of the FDA-approved pharmacopeia

**12 files, +26,172 / −22.**

### 2.1 Scientific intent

The dual-barrier framework had, until this point, been evaluated on held-out test splits and a five-drug clinical benchmark. This commit turns it outward: instead of asking *how accurate is the model*, it asks *what does the model say about every drug a physician can already prescribe*.

That reframing produces two kinds of finding from one screen:

- **Latent CNS penetrants.** Drugs approved for non-CNS indications that the model places in Quadrant 1 ($\text{BBB}^+ / \text{P-gp}^-$) are, in principle, already-safe molecules that should reach the brain. These are repurposing candidates for neuro-oncology, neuroinflammation, and meningeal infection.
- **Efflux-mediated treatment failure.** Marketed neurological and psychiatric drugs that the model places in Quadrant 2 ($\text{BBB}^+ / \text{P-gp}^+$) are molecules that cross the membrane and are then actively pumped back out. This offers a mechanistic hypothesis for clinical non-response in refractory schizophrenia and treatment-resistant depression.

The second finding is the more interesting one, because it is a *negative* prediction about drugs that are already in use, and it is falsifiable.

### 2.2 Data provenance

`scripts/prepare_repurposing_library.py` (87 lines) curates the library:

1. Downloads two files from the **Broad Institute Drug Repurposing Hub**, 2020-03-24 release, from `s3.amazonaws.com/data.clue.io/repurposing/downloads/` — `repurposing_drugs_20200324.txt` (annotations: clinical phase, MoA, target, disease area, indication) and `repurposing_samples_20200324.txt` (structures).
2. Reads both as tab-separated, with `comment='!'` to skip the Hub's metadata preamble.
3. Inner-joins samples to annotations on `pert_iname`, deduplicating to one structure per drug name.
4. Filters to `clinical_phase == 'Launched'` — i.e. drugs with marketing approval, not candidates still in trials.
5. Validates and canonicalises every SMILES through `Chem.MolFromSmiles` then `Chem.MolToSmiles`, discarding anything RDKit cannot parse, then deduplicates on the canonical form.

Output: `data/repurposing/fda_approved_repurposing_compounds.csv`, **2,405 drugs**.

The download is idempotent (skips if the file exists) and sends a browser User-Agent, because the S3 endpoint rejects the default urllib agent.

### 2.3 Code path

`scripts/run_repurposing_screen.py` (409 lines) is the screen. Its defining property is that it **reuses the existing framework rather than re-deriving it** — every scientific convention comes from a module that is already exercised elsewhere:

| Imported from | What it provides |
|---|---|
| `core.features` | `BBB_DESCRIPTOR_NAMES` (6 2D descriptors), `PGP_DESCRIPTOR_NAMES` (those 6 plus 6 3D shape terms), `calculate_2d_descriptors`, `calculate_3d_descriptors`, `calculate_morgan_fingerprint`, `load_scaler_metadata` |
| `core.decision_matrix` | `determine_dual_barrier_quadrant` — the single authority on the four-quadrant convention |
| `explainability.applicability_domain` | `ApplicabilityDomainEstimator` — reference fingerprints for both training sets |

The pipeline runs in five stages.

**(a) Featurisation — parallel, cached.** `process_single_molecule` is dispatched over a `ProcessPoolExecutor` with up to 8 workers. For each molecule it computes a 2048-bit Morgan fingerprint ($r = 2$), the 6 2D physicochemical descriptors, and a single ETKDGv3 conformer from which the 6 `Descriptors3D` shape terms are read. Results are ordered back by original index and pickled to `data/repurposing/repurposing_features_cache.pkl`, so re-runs skip the expensive conformer embedding entirely.

When conformer embedding fails, the 3D block is **zero-filled** and a `has_3d` flag is set false. This is a deliberate degradation rather than a dropped row, and it is worth knowing about: a zero-filled shape vector is not a neutral input to the P-gp model, it is an out-of-distribution one.

**(b) Scaling.** The processed training CSVs store *raw* descriptors, so the saved scaler metadata must be applied at inference time. The script loads `bbb_scaler_metadata.json` and `pgp_scaler_metadata.json` and z-scores each continuous block in bulk with NumPy, guarding zero standard deviations to 1.0.

**(c) Batched inference.** Both feature matrices are assembled as single DataFrames — `FP_0` through `FP_2047` followed by the descriptor columns, in the exact order the saved boosters expect, since XGBoost is position-sensitive — and each model is called **once** with `predict_proba`. The comment in the source calls this a "100x speedup"; vectorising the two model calls is what makes a 2,405-molecule screen a matter of seconds rather than an afternoon.

**(d) Scoring.** Three quantities per drug:

- **Quadrant**, from thresholding both probabilities at 0.5: Q1 $\text{BBB}^+/\text{P-gp}^-$ (optimal CNS), Q2 $\text{BBB}^+/\text{P-gp}^+$ (paradoxical), Q3 $\text{BBB}^-/\text{P-gp}^-$ (peripheral), Q4 $\text{BBB}^-/\text{P-gp}^+$ (restricted).
- **CNS desirability**, $S_{\text{CNS}} = P(\text{BBB}^+) \cdot \left(1 - P(\text{P-gp}^+)\right)$. The product form is the point of the whole framework: a drug needs both to get in *and* to stay in, so neither factor can compensate for the other. A drug with $P(\text{BBB}^+) = 0.99$ and $P(\text{P-gp}^+) = 0.99$ scores $0.0099$, not "excellent on one axis".
- **Applicability domain**, as the maximum Tanimoto similarity to each training set (computed with `BulkTanimotoSimilarity` against the estimator's cached reference fingerprints), labelled `Inside` when both exceed 0.35 and `Borderline` otherwise. This is the honesty valve: it marks predictions made on chemistry the models never saw.

**(e) Cohort extraction.** Two filtered views come out of the full table:

- Non-CNS Quadrant 1, requiring `tanimoto_bbb >= 0.35` — i.e. the model must be inside its domain for the claim to count. Disease area is matched against `neurology|psychiatry` and *excluded*.
- CNS Quadrant 2 — the same disease-area match, *included*, sorted by descending $P(\text{P-gp}^+)$.

### 2.4 Outputs

| Artifact | Rows | Content |
|---|---|---|
| `results/fda_repurposing_dual_barrier_screen.csv` | 2,405 | Full screen: both probabilities, both predictions, quadrant, $S_{\text{CNS}}$, both Tanimoto values, domain status |
| `results/top_cns_repurposing_candidates.csv` | 273 | Non-CNS Q1 hits, in-domain, ranked by $S_{\text{CNS}}$ |
| `results/cns_drugs_efflux_resistance_profiles.csv` | 105 | Marketed neuro/psych drugs in Q2, ranked by efflux probability |
| `figures/repurposing_pharmacopeia_landscape.png` | — | $P(\text{BBB}^+)$ vs $P(\text{P-gp}^+)$ scatter, quadrant-coloured, with Levetiracetam, Loperamide, Diazepam and Caffeine annotated as clinical anchors |
| `figures/repurposing_disease_area_distribution.png` | — | Stacked quadrant percentages across the seven largest therapeutic areas |

The quadrant distribution across the pharmacopeia, recomputed from the committed CSV at `HEAD`:

| Quadrant | Drugs | Share |
|---|---|---|
| Q1 — Optimal CNS ($\text{BBB}^+/\text{P-gp}^-$) | 710 | 29.5% |
| Q2 — Paradoxical ($\text{BBB}^+/\text{P-gp}^+$) | 240 | 10.0% |
| Q3 — Peripheral ($\text{BBB}^-/\text{P-gp}^-$) | 1,064 | 44.2% |
| Q4 — Restricted ($\text{BBB}^-/\text{P-gp}^+$) | 391 | 16.3% |

### 2.5 What entered the manuscript

A new `## 5.` section, with the existing Discussion / Limitations / Reproducibility / Conclusion renumbered 5–8 to 6–9. It carries two tables:

- **Table 5**, the repurposing leads: Thalidomide and Aminoglutethimide (neuro-oncology and brain metastases), Topotecan, Betamethasone and Triamcinolone (vasogenic oedema, microglial modulation), Linezolid and Chloramphenicol (resistant bacterial meningitis), Milnacipran (central neuropathic pain), Metoclopramide, Naloxone.
- **Table 6**, the Q2 efflux-resistance cohort: Flupentixol, Zuclopenthixol, Vilazodone, Everolimus, Haloperidol-decanoate, Lurasidone, Fluphenazine, Aripiprazole-lauroxil, Trifluoperazine, Pimavanserin.

Several Table 5 entries function as positive controls rather than discoveries: Linezolid's high CSF-to-plasma ratio and Chloramphenicol's historical use in meningitis are established clinical facts, so the model recovering them is evidence the screen is calibrated, not a new claim. Everolimus in Table 6 is the sharpest entry, because it is *prescribed* for subependymal giant cell astrocytoma, and the model assigns it a 97.6% efflux probability.

### 2.6 Caveats

- **Topotecan's SMILES in Table 5 was invalid.** Fixed later in `dfbbc5b`.
- **Zero-filled 3D fallback** (§2.3a) silently feeds an out-of-distribution shape vector to the P-gp head.
- Two code smells: a dead line that constructs a throwaway fingerprint via `domain_estimator.bbb_fps[0].__class__(2048)` purely to obtain a type, immediately overwritten; and a reach into the private `_MORGAN_GEN_R2_2048` from `core.features` rather than calling the public `calculate_morgan_fingerprint`.
- The commit tracked the two raw Hub text files (roughly 20,000 lines combined). They remain tracked at `HEAD`, which is defensible for provenance but is part of why the repository is large.
- `.gitignore` gained `*.pkl` in this commit, to keep the feature cache out of version control.

---

## 3. Commit 2 — `7378e3a`: Cryo-EM structural validation against human P-glycoprotein

**51 files, +37,159 / −22.**

### 3.1 Why an orthogonal modality matters

Commit 1's central claim is that the classifier's $P(\text{P-gp}^+)$ tracks something physically real about how a molecule interacts with the efflux pump. But the classifier learned from a binary substrate/non-substrate label table. Nothing in its training forces its probability to correspond to binding at all — it could be exploiting a correlated chemical shortcut (lipophilicity, molecular size) and still score well on a held-out split.

The only way to test that is to compute the interaction by a route that shares **no information** with the model: take the experimental structure of the protein and dock the molecules into it. Molecular docking knows nothing about the training labels, the fingerprints, or the boosters. If the molecules the classifier flags as substrates independently bind the translocation chamber more tightly, that agreement is informative. If they do not, the classifier is a correlation machine.

### 3.2 Receptor preparation

`scripts/run_cryoem_docking_validation.py` (578 lines) provisions its own toolchain, which is the reason it is reproducible from a clean machine:

1. `ensure_vina_installed()` downloads **AutoDock Vina 1.2.7** from the ccsb-scripps GitHub release into `bin/`.
2. `ensure_receptor_prepared()` fetches **`6QEX.pdb`** from `files.rcsb.org` — the 3.5 Å cryo-EM structure of human ABCB1 / P-glycoprotein in the inward-facing conformation, with Taxol bound (Alam et al., *Science* **363**, 753, 2019).
3. Chain A is split out to `pgp_chain_a.pdb` and converted to PDBQT with Meeko's `mk_prepare_receptor`.
4. Ligands are built from SMILES with RDKit and written as PDBQT by `prepare_ligand_pdbqt`.

### 3.3 Docking protocol

The search box is **24 Å** on a side, centred at **(173.33, 166.74, 161.48)** — coordinates taken from the **co-crystallised Taxol (TA1) position in 6QEX itself**. This is the methodological crux: the box is not placed by guesswork, it is placed on the experimentally observed drug-binding site inside the central translocation chamber. Vina runs at `--exhaustiveness 8 --num_modes 5`.

Paclitaxel is included in the compound set specifically as a **re-docking control** — it is the ligand resolved in the structure, so recovering a strong pose for it tests the box placement rather than the hypothesis.

### 3.4 Derived quantities

- **$\Delta G_{\text{bind}}$** (kcal/mol), parsed from Vina's output by `parse_vina_output` along with the pose coordinates.
- **Estimated $K_d$**, as $K_d = \exp\!\left(\Delta G / RT\right)$ with $RT = 0.59248$ kcal/mol at $T = 298.15$ K. This is a thermodynamic restatement of the score, not independent evidence — it inherits every limitation of the scoring function, and reporting nanomolar precision from a docking score overstates what the number can support.
- **Contact residues**, via `find_contact_residues`: every receptor residue with an atom within **3.8 Å** of the docked pose, read out of the receptor PDB by `load_receptor_residues`. The script names the expected chamber-lining residues up front — Phe303, Tyr307, Phe343, Gln725, Phe728, Phe983 — so the recovered contacts can be checked against the structural biology literature rather than merely counted.

### 3.5 Statistics, and what they do and do not establish

Three tests, over **22 compounds** (10 declared Q1, 9 declared Q2, 2 Q3 peripheral controls, plus the Paclitaxel re-docking control):

| Test | Purpose |
|---|---|
| Welch's $t$-test (`equal_var=False`) | Q2 vs Q1 mean binding energy, without assuming equal variances |
| Mann–Whitney $U$, one-sided (`alternative='less'`) | The same comparison without a normality assumption |
| Spearman $\rho$ | Monotone association between ML $P(\text{P-gp}^+)$ and $\Delta G$ |

The commit reports **$p = 0.0035$** and **$\rho = -0.5965$**. Recomputed from the committed CSV at `HEAD`, the mean $\Delta G$ is **−8.84 kcal/mol** for the Q2 group against **−6.91 kcal/mol** for Q1 — roughly 1.9 kcal/mol of separation, in the predicted direction.

The negative $\rho$ is the result that matters, and its sign is easy to misread. $\Delta G$ is negative for favourable binding, so *more negative* means *tighter*. A negative correlation between $P(\text{P-gp}^+)$ and $\Delta G$ therefore means: as the classifier's efflux probability rises, docking energy falls, i.e. the compounds the model calls substrates do bind the chamber more tightly. Two methods that share no inputs agree on the ordering.

What this **does** support: the classifier's efflux probability is not pure chemical happenstance; it co-varies with a structure-based estimate of chamber binding.

What it **does not** support, and should not be claimed:

- **$N = 22$**, with 9 and 10 in the compared groups. A $p$-value of 0.0035 on that sample size is suggestive, not decisive, and the compounds were hand-picked from the extremes of the model's own output — which inflates the separation relative to a random sample.
- **Binding is not transport.** Tight binding in the chamber does not establish that a molecule is effluxed; some tight binders are inhibitors, not substrates. Loperamide and Flupentixol binding well is consistent with the hypothesis but does not demonstrate the mechanism.
- **One receptor conformation.** 6QEX is a single inward-facing state. P-gp is a conformational machine, and docking into one frozen snapshot ignores the transport cycle entirely.
- **Docking scores are crude.** Vina's scoring function carries errors on the order of 1–2 kcal/mol, the same magnitude as the separation being reported. The $K_d$ column should be read as a rank ordering, not as an affinity measurement.

### 3.6 Outputs

`results/cryoem_pgp_docking_validation.csv` (22 rows): compound, SMILES, quadrant, pharmacological class, both ML probabilities, $\Delta G$, estimated $K_d$, $K_d$ in µM, contact residue count, and the key contact list. Plus `figures/cryoem_docking_energy_distribution.png` and `figures/cryoem_docking_vs_ml_correlation.png`, the receptor files, and the per-ligand input and docked pose PDBQTs.

### 3.7 Caveats specific to this commit as written

- **The ML probabilities were hard-coded.** The 22-entry `BENCHMARK_COMPOUNDS` table carried literal `prob_pgp_ml` and `prob_bbb_ml` values typed into the source rather than obtained by calling the models. For a commit whose entire purpose is to correlate two independent methods, a transcribed copy of one of them is a serious defect: it makes the correlation unauditable and lets the two sides drift apart silently. Fixed in `dfbbc5b`, which made the script call the models. **The correction changed two assignments** — once recomputed, Paclitaxel moved from the declared Q2 to Q4, and Aspirin from Q3 to Q1. Both discrepancies are now recorded explicitly in a `quadrant_declared` column alongside the live `quadrant`, which is the right way to carry a disagreement rather than paper over it.
- The commit tracked a 15,086-line `6QEX.pdb` and 44 PDBQT ligand and pose files. The poses were untracked later in `0b788bf`, on the correct reasoning that every run regenerates them and the derived numbers live in the CSV.
- `.gitignore` gained `*.exe` (the Vina binary) and `test_*.pdbqt`.

### 3.8 What entered the manuscript

A new `## 6.` section — chamber architecture and docking protocol (6.1), thermodynamic separation between Q2 and Q1 (6.2), transmembrane helical contact residues (6.3), cross-modality correlation (6.4) — carrying **Table 7**, and renumbering the tail of the paper 6–9 to 7–10.

---

## 4. Commit 3 — `325ef77`: README rewrite

**`README.md` only, +197 / −113.** No code, no data, no results.

### 4.1 What changed

- **Title** extended to include *Cryo-EM Structural Validation*.
- **Renumbered** from 8 sections to 9, with installation and usage pushed down to make room for science.
- **Two mermaid diagrams.** A *Chrono Flow* (new section 2) narrating the research in the order it actually happened, and an end-to-end pipeline `flowchart TD` (new section 3) running data to features to models to decision matrix to screening to docking. Choosing a *chronological* diagram over a purely architectural one is the notable editorial decision: it presents the project as a sequence of questions rather than a static system.
- **A benchmark table** (new section 4) placing the in-domain split beside the **Bemis–Murcko zero-scaffold-overlap** split for both heads, with 1,000-iteration bootstrap 95% confidence intervals on ten metrics each. Showing both regimes side by side is the honest presentation: the scaffold-split balanced accuracy for BBB (0.7605) is visibly worse than the in-domain figure (0.8775), and the table does not hide it.
- **A *Major Scientific Discoveries* section** (new section 5) summarising Discovery 1 (the $N = 2{,}405$ screen) and Discovery 2 (the 6QEX docking).
- **Two new reproduction recipes**, labelled **E** (curate the Hub library, then run the screen) and **F** (run the docking validation), plus a rewritten repository file tree.

### 4.2 Status at HEAD

This README **no longer exists.** Commit `b44c754` ("Consolidate documentation to two public documents, re-derived from source") rewrote it entirely. The current README has different section names (`## The problem`, `## The finding`, `## Results`, `## Install and run`, `## Limitations`), no lettered recipes, and no mermaid diagrams.

The current README does still cover both pieces of work — `### Pharmacopeia screen — 2,405 approved drugs` and `### Cryo-EM structural validation (PDB 6QEX, AutoDock Vina, seeded)` — so commit 3's substance survived even though its structure did not. Read commit 3 for how the project chose to narrate itself in October, not for current instructions.

---

## 5. Reproducing these results today

From inside `Novelty Upgrade/` at `HEAD`:

```bash
# 1. Curate the Broad Drug Repurposing Hub library (needs network access)
python scripts/prepare_repurposing_library.py

# 2. Dual-barrier screen over 2,405 approved drugs
python scripts/run_repurposing_screen.py

# 3. Cryo-EM docking validation into 6QEX (needs network access + AutoDock Vina)
python scripts/run_cryoem_docking_validation.py
```

Practical notes:

- Step 1 downloads from `data.clue.io`; step 3 downloads from `files.rcsb.org` and GitHub. Both are no-ops if the files are already present.
- Step 2 will reuse `data/repurposing/repurposing_features_cache.pkl` if it exists. **Delete that cache** if you have changed anything in `core/features.py`, or you will be scoring stale features.
- Step 3 writes poses to `data/docking/poses/`, which is gitignored; only the CSV and the figures are tracked.

---

## 6. Corrections applied after these commits

Nothing in this table invalidates the framework, but anyone quoting the 20 October numbers should quote the corrected ones.

| Defect introduced in these commits | Fixed in |
|---|---|
| Docking benchmark table carried hard-coded `prob_pgp_ml` / `prob_bbb_ml` instead of calling the models | `dfbbc5b` |
| Topotecan's SMILES (a Table 5 repurposing lead) was invalid | `dfbbc5b` |
| Unscaled descriptors reaching SHAP, producing invalid attributions | `dfbbc5b` |
| Loperamide's benchmark SMILES had one CH2 too many (C30H35ClN2O2 rather than the Hub's C29H33ClN2O2) | `afe832e` |
| Penicillin G lacked the (2S,5R,6R) stereochemistry needed to dock | `afe832e` |
| A blind-validation claim the evidence did not support | `afe832e` (withdrawn) |
| 44 regenerable docking pose files tracked in git | `0b788bf` (untracked) |
| Dead code and two defective evaluation scripts | `760d77a` |
| Title overclaim and a B3DB misattribution | `8af9c2f` |

Known open items at `HEAD`, outside the scope of these three commits but worth noting if you are auditing the same code:

- `explainability/applicability_domain.py` tests `max_tanimoto >= 0.28` while its adjacent comment documents a 0.30–0.40 borderline band.
- The same module's docstring describes its leverage term as a "Z-score Mahalanobis/Euclidean distance"; the code computes an RMS z-distance from the training centroid, which is not a Mahalanobis distance (it ignores descriptor covariance).
- The current README's layout block attributes $S_{\text{CNS}}$ to `core/decision_matrix.py`, but the score is computed in the screening scripts, not there.

---

## 7. How the three commits fit together

Read as one unit, they are a complete argument rather than three features:

1. **`53a1577` makes a claim at scale.** The dual-barrier model, applied to every approved small molecule, identifies 273 repurposing leads and 105 drugs whose clinical failures may be efflux-mediated. On its own, this is a model talking about itself.
2. **`7378e3a` tries to break the claim.** It computes the same interaction from an experimental cryo-EM structure, by a method that shares no inputs with the model, and finds agreement in sign and ordering ($\rho = -0.5965$, $p = 0.0035$, $N = 22$). This is what converts a model output into a result — though a small-$N$, hand-picked-cohort one.
3. **`325ef77` makes both readable.** A chronological diagram, both split regimes shown side by side, and runnable reproduction commands.

The honest summary is that commits 1 and 2 are the scientific contribution, commit 3 is its packaging, and the subsequent audit series (`dfbbc5b`, `afe832e`) is what made the contribution safe to show a reviewer. The arc is worth more than any single commit in it.
