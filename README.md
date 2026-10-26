# Dual-Barrier CNS Drug Prediction

**Every number in this document was re-derived from the files in `Novelty Upgrade/results/`
and is enforced against them by `scripts/verify_doc_consistency.py`.**

---

## The problem

Whether a drug reaches the brain is two questions, not one:

1. Can it **cross** the blood-brain barrier by passive diffusion?
2. Will **P-glycoprotein** — an ATP-powered efflux pump on the blood-facing membrane of
   brain capillary endothelial cells — **pump it straight back out**?

```
Net brain exposure  ∝  passive influx / (active efflux + metabolism)
```

Published BBB benchmarks model only the numerator. **Loperamide** is the proof that the
denominator decides the outcome: a potent µ-opioid agonist, sold over the counter, that
produces no central effect at all because P-gp expels it. **Levetiracetam** is the mirror
image — it crosses *and* evades the pump, so it works in refractory epilepsy.

A model can therefore top a BBB leaderboard while being clinically wrong.

## The finding

We quantified how wrong. B3DB and the Broccatelli P-gp benchmark share **216 molecules with
unambiguous ground truth on both axes** (joined on full InChIKey):

|  | P-gp substrate | P-gp non-substrate | total |
|---|:---:|:---:|:---:|
| **BBB+ (permeant)** | **45** | 95 | **140** |
| **BBB− (non-permeant)** | 19 | 57 | 76 |
| **total** | **64** | **152** | **216** |

> **32.1% of the molecules B3DB labels "permeant" are confirmed P-glycoprotein substrates**
> (45/140, 95% CI [24.5%, 40.6%], Clopper-Pearson).

And the BBB label carries no usable information about efflux status: Cramér's V = **0.064**,
Fisher OR = 1.42 [0.76, 2.67], χ² p = 0.346. *The data are consistent with independence and
the shared variance is negligible; at n = 216 the interval still admits a weak association, so
the defensible statement is that the BBB label is **not a usable proxy** for efflux status.*

**Consequence, measured.** Per-axis ROC-AUC is ≈ 0.94 on **both** barriers, yet:

| Strategy | Correct quadrant | Paradox leakage |
|---|---:|---:|
| Single-axis BBB model (the published status quo) | 55.6% | **93.3%** (42 of 45) |
| Two independent models, composed | 57.9% | 48.9% |
| Shared-encoder multi-task model | 56.0% | 42.2% |

**Paradox leakage** = molecules that are genuinely permeant-but-effluxed which the model
advances as though they reach the brain. A single-axis model leaks **93.3%** of them. Adding
an efflux axis roughly halves that. No metric in current use exposes this.

---

## What is in this repository

| | |
|---|---|
| **Two trained predictors** | BBB permeability and P-gp efflux, Optuna-tuned, leakage-controlled |
| **Explainability layer** | TreeSHAP in log-odds margin space, Morgan bit → chemical fragment, applicability domain, reaction-SMARTS counterfactuals |
| **Pharmacopeia screen** | all 2,405 approved small molecules placed in the four-quadrant matrix |
| **Structural validation** | 22 compounds docked into the cryo-EM structure of human P-gp (PDB 6QEX) |
| **Multi-task GNN arm** | shared GINEConv encoder, two heads, masked loss — and the null result it produced |
| **Decision-aware metrics** | five metrics (M1–M5) that score the *joint* dual-barrier decision |
| **125 automated checks** | four suites asserting that every quoted number matches its source file |

9,674 lines of Python across 41 modules. `Initial Model/` holds the base-paper reference
implementation and is deliberately unmodified.

---

## Results

### Model performance (1,000-resample bootstrap 95% CI)

| | BBB random | BBB scaffold | P-gp random | P-gp scaffold |
|---|:---:|:---:|:---:|:---:|
| ROC-AUC | **0.9522** [0.9410, 0.9629] | 0.9048 [0.8835, 0.9240] | **0.9587** [0.9330, 0.9806] | 0.9631 [0.9359, 0.9846] |
| Accuracy | 0.8856 | 0.8598 | 0.9295 | 0.8838 |
| **Specificity** | **0.8478** | **0.5795** | 0.9189 | 0.9800 |
| MCC | 0.7535 | 0.5718 | 0.8580 | 0.7843 |
| Brier | 0.0847 | 0.0990 | 0.0679 | 0.0811 |

Test sets: 1,171 and 241 molecules. A **Bemis-Murcko scaffold split** shares *zero* molecular
cores with training, so it measures generalisation to genuinely new chemistry. **It is the
honest number.**

**The asymmetry is the result:**

| | ROC-AUC random → scaffold | Specificity |
|---|:---:|:---:|
| **BBB** | 0.9522 → 0.9048 (**−0.047**) | 0.848 → **0.580** |
| **P-gp** | 0.9587 → 0.9631 (**+0.004**) | 0.919 → 0.980 |

The efflux axis transfers to unseen chemotypes; the permeability axis does not. Out of domain
the BBB model defaults to calling almost everything permeant — **precisely the region where
unmodelled efflux does its damage.**

> The scaffold figures come from a fresh, untuned model (300 trees, fixed depth), not the
> Optuna-tuned one, so magnitude partly conflates protocol with tuning. The direction and size
> of the effect are unambiguous.

### Feature ablation (5-fold CV, scaler fitted within each fold)

| Task | Configuration | Features | ROC-AUC |
|---|---|:---:|:---:|
| P-gp | 3D shape only | 6 | 0.8425 ± 0.0243 |
| P-gp | 2D descriptors only | 6 | 0.9185 ± 0.0107 |
| P-gp | 2D + 3D | 12 | 0.9192 ± 0.0131 |
| P-gp | **Morgan + 2D** | 2,054 | **0.9509 ± 0.0070** |
| P-gp | Morgan + 2D + 3D *(deployed)* | 2,060 | 0.9481 ± 0.0076 |
| BBB | 2D descriptors only | 6 | 0.9174 ± 0.0081 |
| BBB | Morgan only | 2,048 | 0.9393 ± 0.0054 |
| BBB | **Morgan + 2D** | 2,054 | **0.9467 ± 0.0035** |

Two readings, both reported:

- **BBB synergy is real** — fingerprints plus descriptors beat either alone.
- **The 3D conformer descriptors do not earn their place.** 0.9481 with, 0.9509 without. Six
  geometric variables alone reach 0.8425, so shape genuinely correlates with cavity fit, but
  2048-bit circular fingerprints already encode it. They are retained for mechanistic
  interpretability, **not accuracy**, and the ETKDGv3 embedding gate costs 14 of 1,219
  molecules (1.1%) to buy nothing. This independently replicates a published negative result.

### What drives each barrier (TreeSHAP, mean |φ|)

| Descriptor | BBB | P-gp | Relationship |
|---|:---:|:---:|---|
| **TPSA** | **1.1825** | 0.1434 | Desolvation filter — BBB-specific |
| **MolLogP** | 0.3992 | **1.3949** | **Antagonistic** |
| MolWt | 0.5366 | 0.5783 | Compound barrier burden |
| HDonors | 0.2711 | 0.0911 | Seelig efflux trigger |
| HAcceptors | 0.1198 | 0.0708 | Polarity modifier |
| RotBonds | 0.1113 | 0.0387 | Induced-fit plasticity |

The priority **inverts** between barriers: polarity governs passive entry, lipophilicity
governs efflux recruitment. **Lipophilicity is antagonistic** — it is what gets a molecule in
and what gets it pumped out. That tension defines the design window: LogP 1.5–3.0,
TPSA < 90 Å², MW < 400 Da, HBD ≤ 1, Rg < 3.8 Å.

### Pharmacopeia screen — 2,405 approved drugs

`S_CNS = P(BBB+) × (1 − P(P-gp+))`. The factorisation is licensed by the measured near-
independence of the two labels above; without that result it would double-count.

| Quadrant | Meaning | Count |
|---|---|---|
| Q1 | BBB+ / P-gp− — reaches the brain | **710 (29.5%)** |
| Q2 | BBB+ / P-gp+ — **paradoxical** | **240 (10.0%)** |
| Q3 | BBB− / P-gp− — peripheral | 1,064 (44.2%) |
| Q4 | BBB− / P-gp+ — doubly restricted | 391 (16.3%) |

→ **273 non-CNS approved drugs in Q1** as repurposing candidates (Thalidomide, Betamethasone,
Linezolid, Milnacipran, Topotecan).
→ **105 marketed neuropsychiatric drugs in Q2**, a structural rationale for treatment
resistance — including **Everolimus** (P(P-gp+) = 0.976), prescribed for subependymal giant
cell astrocytoma, a brain tumour.

### Cryo-EM structural validation (PDB 6QEX, AutoDock Vina, seeded)

| Group | n | ΔG_bind (kcal/mol) |
|---|:---:|:---:|
| Q2 predicted substrates | 9 | **−8.84 ± 0.89** |
| Q1 predicted evaders | 10 | **−6.91 ± 1.30** |

Welch *t* = −3.7933, **p = 1.6 × 10⁻³**; Mann-Whitney *U* = 11.0, p = 3.1 × 10⁻³;
Spearman ρ(P(P-gp+), ΔG) = **−0.5733**, p = 5.3 × 10⁻³.

A ligand-topology model trained only on cell-assay phenotypes agrees with rigid-receptor
binding thermodynamics it was never shown.

> **The separation does not transfer to individual compounds.** Four Q1 evaders bind as
> tightly as the weakest Q2 substrates, and the peripheral control **Penicillin G reaches
> −8.96 kcal/mol** at a predicted efflux probability of 0.020. Rigid-receptor ΔG is a
> population-level signal, not a per-molecule discriminator — consistent with published
> reports, and **precisely why a ligand model adds information the structure does not.**

### Multi-task GNN: a pre-registered null

A 2025 review proposed learning both barriers from one shared encoder. The precondition looked
satisfied — near-independent labels are complementary rather than redundant supervision — and
P-gp, with 848 training molecules against BBB's 5,463, was expected to gain. **The outcome was
declared in writing before the runs.**

| Task | Split | single-task | multi-task | Δ |
|---|---|---:|---:|---:|
| BBB | random | 0.9388 | 0.9404 | +0.0016 |
| BBB | scaffold | 0.7583 | 0.7467 | −0.0116 |
| P-gp | random | 0.9390 | 0.9301 | −0.0089 |
| P-gp | scaffold | 0.8227 | 0.8171 | −0.0056 |

All intervals overlap. Three controls establish this is a result, not a bug:

1. **Plumbing verified.** With the P-gp loss weight set to 0, BBB reproduces single-task
   (0.9411 vs 0.9388) while the untrained P-gp head sits at **0.5626 — chance**.
2. **Confound eliminated.** Upweighting P-gp does not rescue it (0.9295 at ×3, 0.9299 at ×5)
   and progressively **damages** BBB (0.9388 → 0.9310 → 0.9086). The tasks compete for
   encoder capacity — negative transfer, with a dose-response.
3. **It is what the measurement predicts.** Multi-task learning borrows *correlated* signal.
   At Cramér's V = 0.064 there is none to lend. **Had it worked, it would have contradicted
   the central measurement of this project.**

Across 5 paired seeds the deltas (−0.0094 BBB, −0.0027 P-gp) both fall **below the minimum
detectable effect** (0.0187, 0.0429), so the licensed conclusion is *"no effect detected at
this sample size"* — never *"no effect exists."* 14 seeds would settle BBB; P-gp would need 712.

---

## Datasets

| Dataset | File | Rows | Source |
|---|---|---|---|
| **B3DB** — BBB benchmark | `data/B3DB_classification.tsv` | 7,807 | Meng, Xi, Huang & Ayers, *Sci. Data* **8**, 289 (2021) |
| **Broccatelli** — P-gp benchmark | `data/pgp_broccatelli.tab` | 1,219 | Broccatelli et al. 2011, via Therapeutics Data Commons |
| **Drug Repurposing Hub** | `data/repurposing/*.txt` | — | Broad Institute, 2020-03-24 release |
| **Human P-gp** | `data/docking/6QEX.pdb` | — | RCSB PDB; Alam et al., *Science* **363**, 753 (2019) |

Derived, with losses accounted for:

```
7,807 B3DB      → 7,805  (2 SMILES RDKit cannot parse)
1,219 Broccatelli → 1,205 (14 unparseable or ETKDGv3 cannot embed a conformer)
Hub .txt        → 2,405  (clinical_phase == "Launched", RDKit-validated, deduplicated)
B3DB + Broccatelli → 8,801 unique structures (InChIKey join)
                      7,805 BBB-labelled · 1,212 P-gp-labelled · 216 BOTH
```

Feature spaces: **2,054 dims** for BBB (2048-bit Morgan ECFP4 r=2 ‖ 6 physicochemical) and
**2,060** for P-gp (the same ‖ 6 ETKDGv3 shape descriptors).

> The processed CSVs store **raw** descriptor values by design. Every consumer must apply the
> training-derived scaler — use `core.features.load_processed_dataset()`, which does it and
> **raises** if the scaler metadata is missing. Reading them directly feeds raw values to a
> model trained on z-scores.

---

## Install and run

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt
```

The graph-neural-network arm needs `torch_geometric` and falls back to CPU without a GPU.
**Activate the venv before running anything under `gnn/` or `scripts/*multitask*`** — including
`verify_decision_metrics.py` and `verify_multitask_gnn.py`, which import the `gnn` package.
`pytest` currently lives in the system interpreter rather than the venv, so run
`tests/test_explainability.py` outside the venv (or `pip install pytest` into it).

```bash
# Predict one molecule, with the full explainability report
python "Novelty Upgrade/inference.py" --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --explain

# A worked output is committed at results/loperamide_report.{csv,html}

# The five clinical reference drugs
python "Novelty Upgrade/inference.py" --test-benchmark
```

```bash
cd "Novelty Upgrade"

# 1. Featurise          2. Train
python training/bbb_data_preprocessing.py   ; python training/bbb_model_training.py
python training/pgp_data_preprocessing.py   ; python training/pgp_model_training.py

# 3. Validate
python scripts/run_validation.py            # random + scaffold, bootstrap CIs
python scripts/run_ablation_study.py        # feature-subset ablation
python scripts/run_explainability_analysis.py

# 4. Apply
python scripts/run_repurposing_screen.py          # 2,405-drug screen
python scripts/run_cryoem_docking_validation.py   # docking into 6QEX

# 5. Multi-task GNN
python scripts/train_multitask_gnn.py --tasks bbb,pgp --split random
python scripts/run_kfold_multitask.py --tasks bbb,pgp --folds 5 --tag multitask
python scripts/run_metric_analysis.py ; python scripts/run_seed_variance.py

# 6. Verify  — run after anything that changes a saved metric
python scripts/verify_doc_consistency.py        # 45 checks: docs match artifacts
python scripts/verify_decision_metrics.py       # 31 checks: metric definitions
python scripts/verify_multitask_gnn.py          # 29 checks: splits, masking, CUDA
python -m pytest tests/test_explainability.py   #  8 checks: four-quadrant contracts
```

---

## Layout

```
Novelty Upgrade/
  inference.py          single-molecule prediction with --explain
  core/                 features.py (featurisation + scaled loading)
                        decision_matrix.py (four quadrants, S_CNS)
  training/             preprocessing + training, one pair per barrier
  explainability/       TreeSHAP, substructure mapping, applicability domain, counterfactuals
  gnn/                  multi-task GNN: featurizer, model, masked loss, splits, decision metrics
  scripts/              validation, ablation, screening, docking, GNN experiments, 3 verifiers
  tests/                four-quadrant behavioural contracts
  data/ models/ results/ figures/
Initial Model/          base-paper reference implementation (unmodified, not in the pipeline)
                        GCN-BBB on the legacy corpus: 1,890 molecules from the Zhao 2007
                        supporting information, split Adenot (1,621) & Li (269)
docs/VIVA_PREP.md       defence brief: every figure traced to its source file
README.md  RESEARCH_PAPER_MANUSCRIPT.md  requirements.txt
```

---

## Limitations

- **The clinical benchmark drugs are in the training sets.** All five are in B3DB; three are
  also in Broccatelli. Those comparisons are **recall checks, not blind validation**. The
  scaffold split is the only generalisation evidence claimed. (Three showcase drugs *are*
  absent from both at connectivity-block level: Linezolid, Lurasidone, Pimavanserin — but
  three molecules is an illustration, not a validation set.)
- **Only 216 molecules carry both labels**, so every joint analysis rests on a small sample.
  The odds-ratio interval reaches 2.67; Q4 holds ~19 molecules.
- **3D conformer descriptors add nothing** once Morgan fingerprints are present.
- **The dual-barrier framing is not new.** Adenot & Lahana (*J. Chem. Inf. Comput. Sci.* 2004,
  44, 239–248) built a BBB model and a P-gp model and proposed a combined CNS map. The
  contribution here is the **measurement** and its decision-level consequence, not the
  framework.
- **Predicting BBB+ / P-gp− does not guarantee efficacy.** Net exposure also depends on unbound
  brain fraction, plasma protein binding, BCRP and MRP efflux, and metabolic clearance. BCRP is
  the next transporter to add.

---

## Key references

1. **Löscher, W. & Potschka, H.** (2005) Blood-brain barrier active efflux transporters.
   *NeuroRx* **2**(1), 86–98.
2. **Meng, F., Xi, Y., Huang, J. & Ayers, P. W.** (2021) A curated diverse molecular database
   of blood-brain barrier permeability with chemical descriptors. *Sci. Data* **8**, 289.
3. **Broccatelli, F. et al.** (2011) Large-scale classification of P-glycoprotein substrates.
   *J. Med. Chem.* **54**(6), 1740–1751.
4. **Adenot, M. & Lahana, R.** (2004) Blood-brain barrier permeation models. *J. Chem. Inf.
   Comput. Sci.* **44**(1), 239–248.
5. **Alam, A. et al.** (2019) Structural insight into substrate and inhibitor discrimination by
   human P-glycoprotein. *Science* **363**(6428), 753–756.
6. **Seelig, A.** (1998) How does P-glycoprotein recognize its substrates? *Int. J. Clin.
   Pharmacol. Ther.* **36**(1), 50–54.
7. **Lundberg, S. M. et al.** (2020) From local explanations to global understanding with
   explainable AI for trees. *Nat. Mach. Intell.* **2**, 56–67.
8. **Pajouhesh, H. & Lenz, G. R.** (2005) Medicinal chemical properties of successful CNS
   drugs. *NeuroRx* **2**(4), 541–553.
