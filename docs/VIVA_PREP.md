# Viva Preparation Guide

Everything in this repository **except** `Initial Model/` (the base-paper reference
implementation), explained in the order it was built, with every term defined and every
number traced to the file it lives in.

Written for oral defence. **Section 2 is the one to read first** — it settles what the novelty
actually is, and why the multi-task null confirms it rather than undermining it. Section 10 is
the one to re-read last: it lists the weaknesses a sharp examiner will find, and the honest
answer to each.

---

## Contents

1. [The 60-second answer](#1-the-60-second-answer)
2. [**Is the GNN null a problem? No — and here is why**](#2-is-the-gnn-null-a-problem-no--and-here-is-why) ← read this first if you only read one
3. [Vocabulary — every term used in this project](#3-vocabulary)
4. [The datasets](#4-the-datasets)
5. [Chronological build order](#5-chronological-build-order)
6. [End-to-end flow for one molecule](#6-end-to-end-flow-for-one-molecule)
7. [Every result, and how to state it honestly](#7-every-result)
8. [The multi-task GNN and its negative result](#8-the-multi-task-gnn)
9. [Decision-aware metrics M1–M5](#9-decision-aware-metrics)
10. [Weaknesses, and the honest answer to each](#10-weaknesses-and-honest-answers)
11. [Likely viva questions with answers](#11-likely-viva-questions)
12. [Commands](#12-commands)

---

## 1. The 60-second answer

Order matters here. Lead with the system, then the measurement that justifies it, then the null
— never the other way round.

> **"Whether a drug reaches the brain is not one question, it is two.** Can it *get in* across
> the blood-brain barrier, and does the P-glycoprotein pump *throw it back out*. The published
> benchmark literature models only the first. We built a system that models both: an
> explainable dual-barrier predictor reaching **0.9522 ROC-AUC on BBB permeability and 0.9587
> on P-gp efflux**, with bootstrap confidence intervals on every metric, TreeSHAP attributions
> traced back to actual chemical fragments, and an applicability-domain check so it knows when
> it does not know. We then ran it over the **entire approved pharmacopeia — 2,405 drugs** —
> which surfaced **273 non-CNS approved drugs that should reach the brain** as repurposing
> leads, and **105 marketed neuropsychiatric drugs carrying an efflux liability** that offers a
> structural rationale for treatment resistance. We validated the efflux axis independently by
> docking into the **cryo-EM structure of human P-glycoprotein**, where predicted substrates
> bind the translocation cavity significantly more tightly than evaders (p = 1.6×10⁻³).
>
> **Then we measured why the second axis is necessary at all.** On the 216 molecules where the
> two standard benchmarks overlap, **32.1% of the molecules B3DB labels 'permeant' are
> confirmed P-glycoprotein substrates**, and the BBB label carries essentially no information
> about efflux status (Cramér's V = 0.064). So a model can top the B3DB leaderboard while being
> clinically wrong about one in three of its true positives. We built a decision-level metric
> suite to quantify the damage: per-axis ROC-AUC is about 0.94 on **both** barriers, yet only
> **56–58% of molecules land in the correct therapeutic quadrant**, and a single-axis BBB model
> advances **93.3% of paradoxical compounds** as though they reach the brain.
>
> **And we pre-registered and tested the obvious architectural fix** — one shared graph network
> learning both barriers from a single encoder. It returns a **null**, exactly as the
> independence measurement predicts: multi-task learning borrows *correlated* signal, and at
> Cramér's V = 0.064 there is none to lend. We report that null with its mechanism, three
> controls, and a power analysis."

**If the null comes up, go straight to section 2.** It is not a weakness — it is the
experiment that confirms the lead claim, and the argument is airtight.

---

## 2. Is the GNN null a problem? No — and here is why

The multi-task GNN was **never the novelty.** It was the *experiment that tests* the novelty.
If you take one thing from this document, take this section.

### 2.1 The decision gate was pre-registered

The project plan, as it stood (recoverable: `git show 8af9c2f^:docs/RESEARCH_LOG.md` line 1021), **before** the runs happened:

> *"if the P-gp head does not improve, the headline shifts to C2+C3+C4 — the measurement, the
> evaluation, and the attribution — with multi-task reported as a negative result."*

Both outcomes were declared in writing in advance. That is a pre-registered hypothesis with a
pre-declared fallback — what clinical trials are *required* to do, and the opposite of fishing
for a positive result. Nothing was reframed after the fact to rescue a failed experiment.

### 2.2 The null is evidence *for* the lead claim, not against it

Three steps, and they are airtight:

1. **The measurement:** BBB and P-gp labels carry almost no mutual information —
   Cramér's V = 0.064 on 216 molecules with unambiguous ground truth on both axes.
2. **The mechanism of multi-task learning:** it works by borrowing *correlated* signal from an
   auxiliary task. That is the entire principle. No correlation → nothing to borrow.
3. **Therefore the measurement *predicts* the null.** We then ran it, and it failed with the
   predicted mechanism visible.

> **The decisive point: had the multi-task GNN *worked*, it would have contradicted the
> project's own central measurement.** You would then have had two findings that cannot both be
> true. Instead the theory predicts the experiment, and the two findings are one coherent story.

The mechanism was not assumed — it was observed. Upweighting the minority task does **not**
rescue it and progressively **damages** the other:

| P-gp loss weight | P-gp ROC-AUC | BBB ROC-AUC | BBB MCC |
|---|---:|---:|---:|
| single-task baseline | 0.9390 | 0.9388 | 0.7264 |
| 1 (equal) | 0.9301 | 0.9404 | 0.7078 |
| 3 | 0.9295 | 0.9310 | 0.6821 |
| 5 | 0.9299 | 0.9086 | 0.6467 |

**The tasks compete for encoder capacity instead of helping each other.** That is negative
transfer, measured, with a dose-response relationship — not a shrug.

### 2.3 The contribution is three-part, and all three hold

From the build log (`git show 8af9c2f^:docs/RESEARCH_LOG.md`, line 1327):

| | Claim | Status |
|---|---|---|
| **Leg 1** | BBB and P-gp labels are near-independent (Cramér's V = 0.064, n = 216); **32.1% of B3DB "permeant" molecules are confirmed efflux substrates** [24.5, 40.6] | ✅ **The lead claim.** Robust — a prevalence, not a hypothesis test |
| **Leg 2** | **Multi-task graph learning cannot bridge them.** No effect on either task, either split, single-split or pooled K-fold; upweighting causes measurable negative transfer; plumbing verified | ✅ A clean, mechanistically explained null |
| **Leg 3** | **Per-axis metrics hide the decision that matters** — ROC-AUC ≈ 0.94 on both axes, yet 56–58% quadrant accuracy and **93.3% paradox leakage** | ✅ The supporting contribution |

**Leg 2 is a contribution *because* it is a null.** A 2025 review proposed this exact
architecture as promising future work — *"multi-task learning leverages auxiliary signals such
as transporter affinity to guide extraction of disentangled, mechanism-relevant features."*
Nobody had tested it. We tested it on both standard public benchmarks, eliminated the confounds,
and explained the result. The research log names *Journal of Cheminformatics* as the venue for
exactly this.

### 2.4 Two sentences never to say, and the one to say instead

- ❌ **"We built the first joint BBB/efflux GNN and it works."** False, and Adenot & Lahana 2004
  scooped the dual-barrier-framework claim regardless (their PDF is in this repo).
- ❌ **"Multi-task learning is our novel contribution."** Invites the question *"so your main
  thing failed?"* — which is a framing wound, not a scientific one.
- ✅ **"We quantified a blind spot in the field's standard benchmark, and tested whether the
  obvious architectural fix closes it. It doesn't, for the reason the measurement predicts."**

Also true and worth remembering under pressure: **nothing about the GNN null touches the parts
that plainly work.** The 0.9522 / 0.9587 predictors, the explainability layer, the 2,405-drug
screen with its 273 + 105 findings, and the cryo-EM docking all stand independently. The
engineering works; one hypothesis tested with it returned a null. Those are different sentences.

### 2.5 Hostile follow-up drill

**Q: So your main contribution failed?**
No — the contribution is the measurement, and the measurement is what predicted this outcome.
We pre-registered both branches before running anything; the decision gate is in the build log, commit `8af9c2f^`, and in
writing. A shared encoder was the architectural hypothesis the measurement implied *would not*
work, and testing it was the honest thing to do rather than assert it.

**Q: Isn't a negative result just a failed experiment?**
A failed experiment is one where you cannot tell why it failed. We can: no correlated signal to
transfer (V = 0.064), observable negative transfer under upweighting, with a dose-response
curve. A null with a verified mechanism on an architecture a published review recommended is a
result. It tells the next group not to spend six months on it.

**Q: How do I know it's not a bug in your code?**
Three controls. **(1)** The weight-0 sanity run: the multi-task architecture with the P-gp loss
weight set to zero reproduces single-task BBB at **0.9411** (vs 0.9388) while its untrained P-gp
head sits at **0.5626 — chance.** So masking genuinely removes a task and the trunk trains
normally. **(2)** 29/29 verification checks in `verify_multitask_gnn.py`, including an assertion
that a head with no labelled molecules in a batch receives *exactly* zero gradient. **(3)** The
failure is consistent across both splits, both tasks, and three task-weight settings — bugs are
rarely that orderly.

**Q: Why build it at all if you expected it to fail?**
Because an untested prediction is not a result. The measurement implied the null, but implication
is not evidence — and the architecture is precisely the one a 2025 review named as promising, so
leaving it untested would have left the project's central claim open to "but have you tried
multi-task learning?" Now the answer is yes, with numbers.

**Q: Five seeds is not many.**
Agreed, which is why the minimum detectable effect is reported rather than hidden: MDE = 0.01865
for BBB, 0.04295 for P-gp at 80% power. Both measured deltas (−0.0094, −0.0027) fall **below**
it, so the licensed conclusion is *"no effect detected at this sample size"* — never *"no effect
exists."* 14 seeds would settle the BBB ordering; P-gp would need 712, which is itself the
strongest form that null can take.

---

## 3. Vocabulary

Define these before you use them. Examiners interrupt on undefined jargon.

### Biology

| Term | Meaning |
|---|---|
| **BBB** (blood-brain barrier) | A continuous monolayer of brain capillary endothelial cells sealed by tight junctions (claudin-5, occludin), wrapped by pericytes and astrocyte end-feet. It blocks hydrophilic solutes and macromolecules from reaching brain tissue. |
| **Endothelial cell** | The cell lining a blood vessel. At the BBB these cells have almost no gaps between them, unlike peripheral capillaries. |
| **Tight junction** | Protein complex welding adjacent endothelial cells so solutes cannot slip between them. Forces drugs to go *through* the cell membrane (transcellular) rather than around it (paracellular). |
| **Transcellular passive diffusion** | A molecule dissolving into the oily lipid bilayer, crossing it, and emerging on the other side. Driven by lipophilicity and small size. No energy spent. |
| **P-glycoprotein (P-gp)** | A 170 kDa efflux pump on the luminal (blood-facing) membrane of BBB endothelial cells. Also called **ABCB1** or **MDR1**. It grabs lipophilic molecules out of the membrane and pumps them back into blood using ATP. |
| **ABC transporter** | ATP-Binding Cassette transporter family. P-gp is the clinically dominant member at the BBB. Others: **BCRP/ABCG2**, **MRPs/ABCCs**. |
| **Efflux** | Active, energy-consuming transport *out of* a cell or compartment. The opposite of influx. |
| **Substrate** (of P-gp) | A molecule the pump recognises and transports. `P-gp+` in this project. `P-gp−` = non-substrate = "evader". |
| **Translocation cavity** | P-gp's internal chamber, ~6,000 Å³, lined with aromatic (Phe, Tyr) and polar (Gln) residues. Large and flexible, which is why P-gp is *multispecific* — it transports chemically unrelated drugs. |
| **TMD / NBD** | Transmembrane domain (the helices forming the cavity) / nucleotide-binding domain (where ATP binds and hydrolyses, powering the pump). |
| **Paradoxical drug** | A drug that is BBB+ *and* P-gp+: it enters the membrane freely but is pumped straight back out, so it has negligible brain effect. **Loperamide (Imodium)** is the textbook case — a potent opioid with zero central effect. |
| **Pharmacoresistance** | Treatment refractoriness. ~30% of patients with schizophrenia, major depression or epilepsy do not remit despite drug therapy. Elevated endothelial P-gp is one proposed mechanism. |
| **Seelig pharmacophore rules** | Seelig (1998, 2000): P-gp recognises *pairs* of hydrogen-bonding groups separated by fixed spatial distances (2.5 Å or 4.6 Å). More H-bonding units → exponentially faster ATPase activation. Hence `HBD ≤ 1` is a design goal. |
| **Pajouhesh & Lenz rules** | Empirical CNS drug property window: MW < 400, LogP 2–4, TPSA < 90 Å², HBD ≤ 1–2. |
| **Löscher & Potschka (2005)** | The review this project's clinical framing comes from. Source of the Levetiracetam / Loperamide contrast. |
| **f_u,brain** | Unbound (free) drug fraction in brain tissue. Only unbound drug is pharmacologically active, which is why BBB+/P-gp− does not guarantee efficacy. |

### Cheminformatics

| Term | Meaning |
|---|---|
| **SMILES** | Simplified Molecular-Input Line-Entry System. A molecule as an ASCII string: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` is caffeine. Lowercase letters mean aromatic atoms. `@`/`@@` mark stereocentres. |
| **Canonical SMILES** | The one unique SMILES RDKit produces for a structure, so that two spellings of the same molecule compare equal. |
| **InChIKey** | A 27-character hash of a structure (e.g. `RDOIQAHITMMDAJ-...`). Stereo-sensitive and spelling-invariant, which is why this project uses it as the **join key** between the two benchmarks rather than SMILES text. The first 14 characters are the connectivity ("skeleton") block. |
| **RDKit** | The open-source cheminformatics toolkit doing all parsing, descriptor calculation, fingerprinting and 3D embedding here. |
| **Morgan fingerprint / ECFP4** | Extended-Connectivity FingerPrint. For each atom, hash its circular neighbourhood out to radius *r*, then set a bit in a fixed-length vector. Here **radius 2, 2048 bits**. "ECFP4" means diameter 4 = radius 2. Result: a 2048-length binary "barcode" of which substructures are present. |
| **Bit collision** | Two different substructures hashing to the same bit. Unavoidable at 2048 bits; a reason to interpret single-bit attributions cautiously. |
| **Descriptor** | A single computed number describing a molecule. |
| **MolWt** | Molecular weight (Da). Bigger molecules diffuse more slowly and give P-gp more contact points. |
| **LogP / MolLogP** | Logarithm of the octanol/water partition coefficient. The lipophilicity ("greasiness") measure. High LogP helps passive entry **and** recruits P-gp — this is the project's key antagonism. |
| **TPSA** | Topological Polar Surface Area (Å²). Surface contributed by polar atoms (N, O). High TPSA means a thick water shell that must be stripped before entering a lipid membrane — a **desolvation penalty**. |
| **HBD / HBA** | Hydrogen-bond donors / acceptors (counts). |
| **RotBonds** | Rotatable bonds. A flexibility proxy; flexible molecules can adopt an induced-fit pose inside P-gp. |
| **Conformer** | One specific 3D arrangement of a molecule's atoms. A flexible molecule has many. |
| **ETKDGv3** | Experimental-Torsion Knowledge Distance Geometry, version 3. RDKit's 3D conformer generator: it solves a distance-geometry embedding, then corrects torsion angles using knowledge mined from crystal structures. Used here with **fixed seed 42** so the geometry is reproducible. |
| **Radius of gyration (Rg)** | RMS distance of atoms from the centre of mass. A size/bulk measure in Å. |
| **Asphericity, Eccentricity, Spherocity, NPR1, NPR2** | Shape descriptors from the principal moments of inertia. They answer: is this molecule a sphere, a flat disc, or a rod? NPR1/NPR2 are the normalised principal-moment ratios that place a molecule in the standard triangular shape space. |
| **Bemis-Murcko scaffold** | The molecular "core": strip all side chains, keep ring systems and the linkers joining them. Two drugs with the same scaffold are the same chemotype. |
| **Scaffold split** | Partition by scaffold so **no scaffold appears in both train and test**. Tests generalisation to genuinely new chemistry, unlike a random split where a near-twin of every test molecule may sit in train. Always the harder, more honest number. |
| **Tanimoto similarity** | Jaccard index on two fingerprints: shared bits / total bits set. 1.0 = identical fingerprints, 0.0 = nothing in common. |
| **Applicability domain (AD)** | The region of chemical space where a model's predictions can be trusted, because it saw similar molecules in training. Outside it, a confident probability is meaningless. |
| **SMARTS** | A pattern language for substructure matching — regex for molecules. `[O;H1:1]>>[O:1]C` is a *reaction* SMARTS: find a hydroxyl, methylate it. |
| **Bioisostere** | A chemical group swap that preserves biological activity while changing physical properties (e.g. carboxylic acid → amide). |
| **Salt stripping** | Removing counter-ions (e.g. the HCl in a hydrochloride salt) so only the active molecule is modelled. |

### Machine learning

| Term | Meaning |
|---|---|
| **XGBoost** | eXtreme Gradient Boosting. Builds an additive ensemble of decision trees, each new tree fitting the gradient of the loss left by the previous ones. Strong on tabular/sparse data such as fingerprints. |
| **Gradient boosting** | Sequential, error-correcting ensembling. Contrast **bagging** (random forest), where trees are independent. |
| **`scale_pos_weight`** | XGBoost's class-imbalance knob: multiplies the loss on positive examples. Set to `n_negative / n_positive` (1.91 for BBB, 0.74 for P-gp). |
| **`tree_method='hist'`** | Bins continuous features into histograms before splitting. Much faster on 2,000+ features. |
| **Optuna** | Hyperparameter optimisation library. Used here for 50 trials. |
| **TPE** | Tree-structured Parzen Estimator — Optuna's Bayesian search. It models P(hyperparameters \| good score) and samples where improvement is likely, instead of scanning a grid. |
| **Early stopping** | Halt training when a *validation* metric stops improving, to avoid overfitting. 30 rounds during search, 50 for the final BBB model, 20 epochs for the GNN. |
| **StandardScaler / z-score** | `z = (x − μ) / σ`. Here μ and σ are computed **on the training partition only** and serialised to `models/*_scaler_metadata.json`. Fingerprint bits are left alone — they are already 0/1. |
| **Data leakage** | Any path by which test information reaches training. The subtle one here is *scaler leakage*: fitting μ and σ over the whole dataset leaks the test distribution. |
| **GNN** | Graph Neural Network. A molecule is a graph: atoms = nodes, bonds = edges. |
| **Message passing** | Each layer, every atom aggregates feature vectors from its bonded neighbours and updates its own. After *k* layers an atom's vector encodes its *k*-hop neighbourhood. Three layers ≈ a functional group. |
| **GINEConv** | Graph Isomorphism Network with Edge features. GIN is provably as discriminative as the Weisfeiler-Lehman graph isomorphism test — the strongest guarantee in this family. The "E" variant lets **bond** features into the message, which matters because an aromatic bond and a single bond mean different chemistry. |
| **Global mean pooling / readout** | Average all atom vectors into one fixed-length vector for the whole molecule, so a classifier can consume it regardless of molecule size. |
| **BatchNorm** | Normalises activations per mini-batch; stabilises and speeds up training. |
| **Dropout** | Randomly zero a fraction of activations during training (p = 0.30 here) to prevent co-adaptation and overfit. |
| **AdamW** | Adam optimiser with *decoupled* weight decay — the regularisation does not get rescaled by the adaptive learning rate. lr = 5e-4, weight_decay = 5e-4. |
| **Gradient clipping** | Cap the gradient norm (at 2.0) so one pathological batch cannot blow up the weights. |
| **ReduceLROnPlateau** | Halve the learning rate when the monitored metric stops improving (factor 0.5, patience 8). |
| **Multi-task learning** | One shared encoder, several task-specific heads. The hope is that the auxiliary task's gradient teaches the encoder features that help the main task. Works when tasks are **correlated**. |
| **Masked loss** | Compute each head's loss over only the molecules that actually carry that head's label, normalised by that head's own count. Essential here: ~97.5% of molecules have exactly one of the two labels. |
| **One-hot encoding** | A categorical value as a vector with a single 1. Used in the GNN's `encoded` feature mode so that chlorine (Z=17) is not treated as numerically "17× carbon". |
| **TreeSHAP** | Exact SHapley Additive exPlanations for tree ensembles. Shapley values come from cooperative game theory: the unique attribution satisfying efficiency, symmetry, dummy and additivity. For a general model, exact Shapley values are exponential; TreeSHAP exploits tree structure to compute them in polynomial time. |
| **Log-odds margin space** | The raw additive output of the tree ensemble, *before* the sigmoid. Attributions are computed here because the decomposition `f(x) = φ₀ + Σφᵢ` is exactly additive only in margin space. Probability is recovered as `P = 1/(1 + e^{−f(x)})`. |
| **φ₀ (base value / expected value)** | The model's average margin over the training background. **−0.3121 for BBB, +0.1519 for P-gp**. Every attribution is a deviation from this. |
| **Counterfactual** | "What minimal change would flip the prediction?" Here, chemically *valid* changes only — real medicinal-chemistry transformations, not arbitrary feature perturbations. |

### Statistics

| Term | Meaning |
|---|---|
| **ROC-AUC** | Area under the receiver-operating-characteristic curve. Probability that a random positive is ranked above a random negative. Threshold-free. 0.5 = chance. |
| **PR-AUC / average precision** | Area under the precision-recall curve. More informative than ROC-AUC under class imbalance, because it ignores true negatives. |
| **Sensitivity / recall** | TP / (TP + FN). Of the real positives, how many did we catch. |
| **Specificity** | TN / (TN + FP). Of the real negatives, how many did we correctly reject. **The metric that collapses on the BBB scaffold split: 0.848 → 0.580.** |
| **Balanced accuracy** | Mean of sensitivity and specificity. Immune to class imbalance, unlike plain accuracy. |
| **MCC** | Matthews Correlation Coefficient. Correlation between predicted and true labels using all four confusion-matrix cells; −1 to +1. The most honest single number for imbalanced binary problems. |
| **Brier score** | Mean squared error of predicted probabilities. Measures **calibration** — whether a stated 70% really happens 70% of the time. Lower is better. |
| **Bootstrap CI** | Resample the test set with replacement 1,000 times, recompute the metric each time, report the 2.5th and 97.5th percentiles. Gives an interval without assuming a distribution. |
| **Clopper-Pearson interval** | Exact binomial confidence interval. Used for the 32.1% prevalence claim. |
| **χ² / Fisher exact test** | Tests of association in a 2×2 table. Fisher is exact and preferred at small counts. |
| **Odds ratio (OR)** | Ratio of odds between two groups. OR = 1 means no association. Here OR = 1.42, CI [0.76, 2.67] — the interval crosses 1, so no detectable association. |
| **Cramér's V** | Effect size for a contingency table, 0 to 1. **V = 0.064** means the two labels share almost no information. |
| **Cohen's κ** | Agreement corrected for chance agreement. Used for the 4-class quadrant agreement (M2). |
| **Welch's t-test** | Two-sample t-test that does *not* assume equal variances. Correct here since Q1 and Q2 have different ΔG spreads (1.30 vs 0.89). |
| **Mann-Whitney U** | Non-parametric rank test. Reported alongside the t-test as a distribution-free confirmation. |
| **Spearman ρ** | Rank correlation. Monotonic rather than linear; robust to outliers. |
| **MDE** | Minimum Detectable Effect. The smallest true difference a study with this sample size and variance could detect at 80% power. **If your measured effect is below the MDE, the correct conclusion is "no effect detected", never "no effect exists."** |
| **P(A > B)** | Over all cross-seed pairings, the fraction where method A beats method B. Bouthillier et al. recommend reporting this instead of comparing two single numbers; their decision threshold is 0.75. |
| **Paired test** | Both methods evaluated under the same seed, so the seed's effect cancels. This is why the same seed drives both the split and the initialisation in `run_seed_variance.py`. |

### Structural biology / docking

| Term | Meaning |
|---|---|
| **Cryo-EM** | Cryogenic electron microscopy. Flash-freeze the protein and image it with electrons. Unlike X-ray crystallography it needs no crystal, so it works on membrane proteins such as P-gp. |
| **PDB** | Protein Data Bank. **6QEX** = human P-gp in lipid nanodiscs, 3.5 Å, inward-facing, with Paclitaxel bound (Alam et al., *Science* 2019). |
| **Resolution (3.5 Å)** | How finely detail is resolved. 3.5 Å shows backbone and side-chain placement but not individual water molecules — a real limit on docking precision, worth volunteering. |
| **Inward-facing conformation** | The pre-transport state, cavity open to the cytoplasm/inner membrane leaflet. The right state for modelling substrate *capture*. |
| **Chain A** | One polypeptide chain in the PDB file. Human P-gp is a single chain; chain A is extracted and everything else (waters, lipids, other chains) discarded. |
| **PDBQT** | AutoDock's format: PDB plus partial charges (**Q**) and rotatable-bond torsion tree (**T**). |
| **Meeko** | The tool that converts an RDKit molecule or a PDB into PDBQT, assigning Gasteiger partial charges and defining torsions. |
| **AutoDock Vina** | The docking program (v1.2.7 here). Samples ligand poses inside a box and scores them with an empirical free-energy function. |
| **Docking** | Predicting the bound pose and affinity of a small molecule in a protein site. |
| **Rigid receptor** | The protein is held fixed; only the ligand moves. Fast, but it cannot model P-gp's induced fit — a stated limitation. |
| **Grid box** | The search volume. Here a 24 Å cube centred on (173.33, 166.74, 161.48), the co-crystallised Paclitaxel site. |
| **Exhaustiveness** | How hard Vina searches (8 = default). Higher is more thorough and slower. |
| **ΔG_bind** | Predicted binding free energy, kcal/mol. **More negative = tighter binding.** |
| **Kd** | Dissociation constant. `Kd = exp(ΔG / RT)` with RT = 0.59248 kcal/mol at 298.15 K. Lower Kd = tighter. nM is tight, μM is weak. |
| **π-π stacking** | Attraction between aromatic rings, face-to-face or edge-to-face. The dominant interaction in P-gp's Phe-rich cavity. |
| **MMFF94** | Merck Molecular Force Field. Used to energy-minimise the ligand's 3D geometry before docking. |

---

## 4. The datasets

### 4.1 Raw inputs

| Dataset | File | Rows | Source | Columns used |
|---|---|---|---|---|
| **B3DB** — BBB benchmark | `data/B3DB_classification.tsv` | 7,807 | B3DB v1 (Meng et al., *Sci. Data* 2021) | `SMILES`, `BBB+/BBB-`, `logBB` |
| **Broccatelli** — P-gp benchmark | `data/pgp_broccatelli.tab` | 1,219 | Broccatelli et al. 2011, retrieved via **TDC** (Therapeutics Data Commons) | `Drug` (SMILES), `Y` (1 = substrate) |
| **Drug Repurposing Hub** | `data/repurposing/repurposing_{drugs,samples}_20200324.txt` | — | Broad Institute, 2020-03-24 release | `pert_iname`, `smiles`, `clinical_phase`, `moa`, `target`, `disease_area`, `indication` |
| **Human P-gp structure** | `data/docking/6QEX.pdb` | — | RCSB PDB | chain A ATOM records |

### 4.2 Derived files, and why the counts shrink

```
B3DB_classification.tsv          7,807 rows
  └─ bbb_data_preprocessing.py   → 2 SMILES RDKit cannot parse are dropped
       processed_b3db_data_advanced.csv   7,805 × (SMILES, Label, FP_0..FP_2047, 6 descriptors)

pgp_broccatelli.tab              1,219 rows
  └─ pgp_data_preprocessing.py   → 14 dropped: unparseable, or ETKDGv3 fails to embed a 3D conformer
       processed_pgp_data.csv            1,205 × (SMILES, Label, FP_0..FP_2047, 12 descriptors)

repurposing_{drugs,samples}.txt
  └─ prepare_repurposing_library.py  → merge on pert_iname, keep clinical_phase == "Launched",
                                        validate with RDKit, deduplicate on canonical SMILES
       fda_approved_repurposing_compounds.csv    2,405 approved small molecules

B3DB + Broccatelli
  └─ gnn/data_assembly.py        → canonicalise, InChIKey join, drop structures whose
                                    duplicate rows disagree on the label
       merged corpus (in memory)         8,801 unique structures
                                          ~7,805 carry a BBB label
                                          ~1,212 carry a P-gp label
                                             216 carry BOTH  ← everything joint rests on these
```

**Class balance.** B3DB: 4,956 BBB+ (63.5%) / 2,849 BBB− (36.5%). Broccatelli: 650 substrates
(53.9%) / 555 non-substrates (46.1%).

### 4.3 The 216-molecule overlap — the heart of the project

Joined by **full InChIKey**, so spelling, tautomer and salt differences cannot cause a false
miss. All 216 carry one unanimous label per axis.

|  | P-gp substrate | P-gp non-substrate | total |
|---|:---:|:---:|:---:|
| **BBB+ (permeant)** | **45** | 95 | **140** |
| **BBB− (non-permeant)** | 19 | 57 | 76 |
| **total** | **64** | **152** | **216** |

Two claims, at two strengths:

- **Primary (a prevalence — needs no hypothesis test): 32.1% of B3DB BBB+ molecules in this
  set are confirmed P-gp substrates** — 45/140, 95% CI [24.5%, 40.6%], Clopper-Pearson.
- **Secondary (an association test, with its limits stated):** no detectable association.
  χ² = 0.887, p = 0.346; Fisher OR = 1.42, 95% CI [0.76, 2.67]; Cramér's V = 0.064.
  P(P-gp+ | BBB+) = 32.1% vs P(P-gp+ | BBB−) = 25.0%.

**Never say "the labels are statistically independent."** At n = 216 the interval still admits
an OR of 2.67. The defensible phrasing: *the BBB label is not a usable proxy for efflux status,
and any residual association is bounded as weak.*

**Also worth knowing** (it shows depth): B3DB enumerates stereoisomers heavily — 7,805 distinct
molecules collapse to 4,022 connectivity blocks, and **130 blocks carry internally conflicting
BBB labels**. That matters to anyone deduplicating B3DB.

### 4.4 Two feature spaces

| Model | Dimensions | Composition |
|---|---|---|
| **M_BBB** | **2,054** | 2048-bit Morgan ECFP4 ‖ 6 2D descriptors |
| **M_Pgp** | **2,060** | 2048-bit Morgan ECFP4 ‖ 6 2D descriptors ‖ 6 ETKDGv3 3D shape descriptors |

2D set: `MolWt, LogP, TPSA, HDonors, HAcceptors, RotBonds`
3D set: `RadiusOfGyration, Asphericity, Eccentricity, SpherocityIndex, NPR1, NPR2`

> **The processed CSVs store RAW descriptor values on purpose.** Any consumer must apply the
> training-derived scaler. Use `core.features.load_processed_dataset()`, which does it and
> **raises** if the scaler metadata is missing. This is not a style preference — three call
> sites once forgot, which silently fed MolWt ≈ 384 to a model trained on z-scores and
> invalidated an entire SHAP analysis. Failing loudly was the fix.

---

## 5. Chronological build order

### Stage 0 — `Initial Model/` (base paper, out of scope, kept unmodified)

Reference implementation of GCN-BBB on the legacy Adenot/Li corpus: 1,890 molecules from the
Zhao 2007 supporting information, split Adenot (1,621) + Li (269). Single-task BBB only, no
efflux awareness. Retained for provenance and as the baseline the rest improves on. **It is not
part of any pipeline below.**

Known defects in that arm (recorded, deliberately not fixed, because fixing a base-paper
reference would destroy the provenance): `train_gnn.py` silently discards the 269 Li molecules,
so all its metrics are Adenot-only; `predict.py` cannot load its own default checkpoint
(hardcodes `HIDDEN_CHANNELS = 256` against a 128-dim artifact); no metrics are persisted
anywhere. Knowing these is itself a good answer to "why did you rebuild rather than extend?"

### Stage 1 — The diagnosis: one endpoint is not enough

The driving observation: `Net brain exposure ∝ passive influx / (active efflux + metabolism)`.
A single BBB classifier models only the numerator. Loperamide is the existence proof that the
denominator decides the clinical outcome. **Everything after this is a response to that.**

### Stage 2 — Featurisation (`core/features.py`)

The single canonical path from SMILES to model input. One module, so the training pipeline, the
inference CLI, the screen and the docking study cannot drift apart.

- `calculate_2d_descriptors` — the 6 physicochemical numbers.
- `calculate_3d_descriptors` — `AddHs` → ETKDGv3 embed (seed 42) → on failure retry with
  `maxAttempts=100` → compute the 6 shape descriptors. Returns `None` if embedding fails, so
  callers must handle it rather than silently receiving garbage.
- `calculate_morgan_fingerprint` — radius 2, 2048 bits, via a module-level cached generator.
- `prepare_dataframe` — z-scores the continuous block with the saved μ/σ and assembles a
  single-row DataFrame **in exactly the training column order**. XGBoost boosters are
  position-sensitive; reordering corrupts inference silently.
- `load_processed_dataset` — the only sanctioned way to read a processed CSV, scaling included.

### Stage 3 — Preprocessing and training (`training/`)

Two symmetric pairs, one per barrier.

**BBB** (`bbb_data_preprocessing.py` → `bbb_model_training.py`):
- Strict **70 / 15 / 15** train / validation / test, stratified, `random_state=42`.
- Test set isolated **before** anything else. Scaler fitted on train only, serialised to
  `models/bbb_scaler_metadata.json`.
- **Optuna, 50 TPE trials**, maximising **validation F1**, tuning `learning_rate` (log
  0.01–0.3), `max_depth` (3–10), `subsample`, `colsample_bytree` (0.5–1.0),
  `min_child_weight` (1–10), `gamma`, `reg_alpha`, `reg_lambda`. Early stopping 30 rounds on
  validation. **The test set is never touched during search.**
- Final model refit with the best parameters, `n_estimators=2000`, early stopping 50,
  `scale_pos_weight = 1.91`.
- Then — and only then — the held-out test set is scored once.

**P-gp** (`pgp_data_preprocessing.py` → `pgp_model_training.py`):
- 1,205 molecules × 2,060 features. Stratified **80 / 20**.
- Fixed hyperparameters (no Optuna): 300 trees, lr 0.05, depth 5, subsample 0.8,
  colsample 0.8, `scale_pos_weight = 0.74`. The dataset is small enough that a 50-trial
  search on 241 test molecules would mostly fit noise.

### Stage 4 — Explainability (`explainability/`)

| Module | What it does |
|---|---|
| `shap_explainer.py` | TreeSHAP in **log-odds margin space**: `f(x) = φ₀ + Σφᵢ`, `P = σ(f(x))`. Separates continuous-descriptor attributions from *active* fingerprint bits, and splits the top features into **supporting** vs **opposing** the predicted class. Includes a compatibility patch for XGBoost 2.0+ writing `base_score` as the bracketed string `'[5E-1]'`, plus a gain-weighted fallback if TreeExplainer cannot initialise. |
| `substructure_mapper.py` | Turns an attributed bit back into chemistry. For bit *k*, read RDKit's `bitInfo` to get (centre atom, radius), call `FindAtomEnvironmentOfRadiusN` + `PathToSubmol`, emit the fragment SMILES and a functional-group tag. Without this, "FP_1829 = +0.41" is uninterpretable. |
| `applicability_domain.py` | Two independent checks. **Max Tanimoto** against the reference library (bulk-computed). **Descriptor leverage** = `sqrt(mean(z²))`, distance from the training centroid in standardised space. Inside domain iff `T_max ≥ 0.40 AND leverage ≤ 2.5`; borderline if `T_max ≥ 0.28 AND leverage ≤ 3.5`; outside otherwise. Plus an uncertainty score `1 − |p − 0.5| / 0.5`, which is 1.0 at the decision boundary and 0.0 at a confident extreme. |
| `counterfactual.py` | Eight medicinal-chemistry **reaction SMARTS** (O-methylation, N-methylation, esterification, acid→amide, aromatic fluorination, O-demethylation, N-demethylation, ketone deoxygenation). Each product is sanitised, required to keep **Tanimoto ≥ 0.50** to the parent, re-featurised *including a fresh 3D conformer*, re-predicted, and kept only if it flips the classification in the desired direction. |
| `shared_properties.py` | Compares the same descriptor's attribution across the two tasks and labels the relationship (antagonistic, compound barrier, efflux trigger…). |
| `report_generator.py` | HTML / CSV / JSON export of a full single-molecule report. |

### Stage 5 — Validation (`scripts/run_validation.py`)

Two protocols, both with **1,000-resample bootstrap 95% CIs** on ten metrics.

1. **Random held-out** — reload the saved booster, reproduce the exact training split
   (`random_state=42`), scale with the saved μ/σ, score once.
2. **Bemis-Murcko scaffold split** — cluster by Murcko scaffold, fill partitions by cluster so
   scaffold overlap is **zero**, refit a scaler on the scaffold-train block, and train a
   **fresh model** for this protocol.

> **Say this before you are asked:** the scaffold number comes from a fresh, *untuned* model
> (300 trees, depth 6, fixed), not the Optuna-tuned one. So random-vs-scaffold is not strictly
> apples-to-apples. The direction of the effect is unambiguous and large, but the magnitude
> partly conflates protocol with tuning. The honest fix is to re-tune inside the scaffold
> protocol.

### Stage 6 — Ablation (`scripts/run_ablation_study.py`)

Stratified 5-fold CV, **scaler fitted within each fold**, same XGBoost settings across all
configurations so only the feature set varies. P-gp: 5 configurations; BBB: 3. This is the
experiment that produced the project's most uncomfortable — and most credible — finding: the
3D descriptors in the title add nothing once Morgan fingerprints are present.

### Stage 7 — Global explainability (`scripts/run_explainability_analysis.py`)

TreeSHAP over held-out test subsets (500 BBB, 250 P-gp), mean |SHAP| per feature, cross-task
Spearman correlation, four publication figures. Reads data **through `load_processed_dataset`**,
which is the fix for the unscaled-SHAP defect.

### Stage 8 — Pharmacopeia screen (`scripts/run_repurposing_screen.py`)

The translational pay-off.

1. Load the 2,405 approved drugs.
2. Featurise in parallel (`ProcessPoolExecutor`, up to 8 workers — ETKDGv3 is the bottleneck),
   cached to a pickle so re-runs are instant.
3. Batch-predict both barriers as matrices, not row by row.
4. Assign quadrants and compute **S_CNS = P(BBB+) × (1 − P(P-gp+))**.
5. Bulk-Tanimoto applicability domain against both reference libraries.
6. Two filtered extracts: **non-CNS drugs in Q1** (`disease_area` excludes
   neurology/psychiatry, `T_max ≥ 0.35`) → repurposing leads; **CNS drugs in Q2** → efflux
   liability.

### Stage 9 — Cryo-EM docking validation (`scripts/run_cryoem_docking_validation.py`)

An *orthogonal* modality: ligand-topology ML versus receptor-structure physics.

1. Download Vina 1.2.7 and PDB 6QEX if absent.
2. Extract chain A, convert to PDBQT with Meeko, box 24 Å centred on the Paclitaxel site.
3. **`resolve_benchmark_smiles()`** — replace the hand-typed benchmark SMILES with curated
   structures (Broad library first, clinical reference set second, hand-typed last) and **print
   every substitution**. Six compounds had a genuinely different *skeleton*; four more differed
   in stereochemistry, which still changes a 3D pose.
4. **`compute_ml_probabilities()`** — compute P(BBB+) and P(P-gp+) from the saved models at
   runtime, and **derive** the quadrant via the shared `quadrant_id` convention, keeping the
   hand-written label as `quadrant_declared` so disagreements are *reported*, not hidden. These
   were previously 23 hard-coded literals, which made the "cross-modality correlation" a
   correlation against numbers typed into the file.
5. Ligand prep: `AddHs` → ETKDGv3 (seed 42) → MMFF94 minimise → Meeko → PDBQT.
6. Dock with **`--seed 42`**. Vina seeds from the clock by default, so without this no reported
   ΔG was reproducible and the manuscript could not legitimately quote a number.
7. Parse the best affinity, derive Kd, find contact residues within 3.8 Å by brute-force
   distance, run Welch's t-test, Mann-Whitney U and Spearman ρ.

### Stage 10 — Multi-task GNN (`gnn/`, `scripts/train_multitask_gnn.py` and friends)

A deliberate, **pre-registered** test of the obvious next idea. Full detail in §8; why its null
is a contribution rather than a failure is in §2.

### Stage 11 — Decision-aware metrics and statistical rigour

`gnn/decision_metrics.py` (M1–M5), `scripts/run_kfold_multitask.py` (out-of-fold predictions
for all 216 dual-labelled molecules), `scripts/run_seed_variance.py` (5 seeds, MDE, P(A>B)).
Full detail in §9.

### Stage 12 — Audit and verification

| Suite | Checks | What it guards |
|---|---|---|
| `scripts/verify_doc_consistency.py` | 57 | Every number quoted in prose matches the artifact it came from |
| `scripts/verify_decision_metrics.py` | 31 | Metric definitions, quadrant convention consistency |
| `scripts/verify_multitask_gnn.py` | 29 | Splits, masking, zero-gradient behaviour, CUDA |
| `tests/test_explainability.py` | 8 | Four-quadrant behavioural contracts |

Three manuscript defects existed purely because figures were copied between documents instead
of re-read from source. Hence the rule: **run `verify_doc_consistency.py` after anything that
changes a saved metric.**

Defects found and fixed in this phase, each worth citing as evidence of rigour:
- **Unscaled SHAP** — global attributions computed on raw descriptors. Fixed by routing all
  reads through `load_processed_dataset`, which raises rather than returning unscaled data.
  Effect: BBB TPSA attribution 0.9875 → 1.1825, MolWt 0.6955 → 0.5366. Conclusions unchanged.
- **Hard-coded docking probabilities** — 23 literals replaced by live model output.
- **Invalid Topotecan SMILES**, and a **Loperamide SMILES with one extra CH₂**
  (C₃₀H₃₅ClN₂O₂, MW 491.08 instead of C₂₉H₃₃ClN₂O₂, MW 477.05).
- **A withdrawn claim.** The wrong Loperamide structure is absent from both databases, which
  had been reported as "blind validation". With the correct structure Loperamide is present in
  both. The blind-validation claim was **retracted in the manuscript**, and Table 4 is now
  labelled a recall check.
- **PyG silently corrupting a row pointer.** Graphs carried `corpus_index`; PyG's default
  `Data.__inc__` increments *any* attribute whose key contains `index` by the graph's node
  count during collation. Predictions came out offset by cumulative node counts. Renamed to
  `corpus_row` with a cache-version bump. Had it not crashed, every saved prediction would have
  been attributed to the wrong molecule.
- **Metrics stored at 4 dp** made exact delta arithmetic impossible when the effects under
  study are of order 1e-3. Now stored at 6 dp, formatted to 4 only for display.

---

## 6. End-to-end flow for one molecule

```
python "Novelty Upgrade/inference.py" --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --explain
```

```
                       SMILES string  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"   (caffeine)
                                 │
                       Chem.MolFromSmiles  →  RDKit Mol      (None → exit 1, no silent guess)
                                 │
          ┌──────────────────────┴──────────────────────┐
          ▼                                             ▼
   extract_features_bbb                          extract_features_pgp
   ├─ Morgan ECFP4 r=2            [1, 2048]      ├─ same fingerprint              [1, 2048]
   └─ 6 2D descriptors            [1, 6]         ├─ same 6 2D descriptors         [1, 6]
                                                 └─ AddHs → ETKDGv3(seed 42)
                                                    → 6 shape descriptors         [1, 6]
          │                                             │
   prepare_dataframe                             prepare_dataframe
   z-score the 6 with bbb μ/σ                    z-score the 12 with pgp μ/σ
   assemble in training column order             assemble in training column order
          ▼                                             ▼
     X_bbb  [1, 2054]                              X_pgp  [1, 2060]
          │                                             │
   bbb_xgboost.json                              pgp_xgboost.json
   predict_proba → P(BBB+)                       predict_proba → P(P-gp+)
          │                                             │
   pred = 1 if P > 0.5                           pred = 1 if P > 0.5
          └──────────────────────┬──────────────────────┘
                                 ▼
                  determine_dual_barrier_quadrant(bbb_pred, pgp_pred)
                  ┌────────────────────────────────────────────┐
                  │ Q1  BBB+ / P-gp−   Optimal CNS candidate   │
                  │ Q2  BBB+ / P-gp+   PARADOXICAL, efflux risk│
                  │ Q3  BBB− / P-gp−   Peripheral agent        │
                  │ Q4  BBB− / P-gp+   Doubly restricted       │
                  └────────────────────────────────────────────┘
                  S_CNS = P(BBB+) × (1 − P(P-gp+))
                                 │
            ┌────────────────────┼────────────────────┬──────────────────────┐
            ▼                    ▼                    ▼                      ▼
      TreeSHAP ×2         Applicability        Shared-property        Counterfactuals
   f(x) = φ₀ + Σφᵢ      max Tanimoto +        cross-task table        8 reaction SMARTS
   per-feature φ        leverage + uncertainty  per descriptor        Tanimoto ≥ 0.50
   active bits →
   fragment SMILES
            └────────────────────┴────────────────────┴──────────────────────┘
                                 ▼
                 Terminal report (6 sections) + optional HTML / CSV / JSON
```

A worked example is committed at `results/loperamide_report.{csv,html}`.

The GNN arm is a separate path: `SMILES → smiles_to_data() → Data(x[V,66], edge_index[2,2E],
edge_attr[2E,14]) → 3× GINEConv → global_mean_pool → h[B,256] → two heads → {bbb: [B,2],
pgp: [B,2]}`.

---

## 7. Every result

### 7.1 Headline performance (`results/validation_metrics_summary.csv`)

| Metric | BBB random (15% held out) | BBB scaffold | P-gp random (20%) | P-gp scaffold |
|---|:---:|:---:|:---:|:---:|
| Accuracy | 0.8856 [0.8659–0.9035] | 0.8598 | 0.9295 [0.8963–0.9585] | 0.8838 |
| Balanced accuracy | 0.8775 | 0.7605 | 0.9287 | 0.8978 |
| Precision | 0.9122 | 0.8849 | 0.9313 | 0.9829 |
| Recall | 0.9073 | 0.9415 | 0.9385 | 0.8156 |
| **Specificity** | **0.8478** | **0.5795** | 0.9189 | 0.9800 |
| F1 | 0.9097 | 0.9123 | 0.9349 | 0.8915 |
| **ROC-AUC** | **0.9522** [0.9410–0.9629] | **0.9048** [0.8835–0.9240] | **0.9587** [0.9330–0.9806] | **0.9631** [0.9359–0.9846] |
| PR-AUC | 0.9724 | 0.9668 | 0.9685 | 0.9801 |
| MCC | 0.7535 | 0.5718 | 0.8580 | 0.7843 |
| Brier | 0.0847 | 0.0990 | 0.0679 | 0.0811 |

Test sizes: BBB 1,171 molecules; P-gp 241.

**The asymmetry is the finding, not the 0.95.**

| | ROC-AUC random → scaffold | Specificity random → scaffold |
|---|:---:|:---:|
| **BBB** | 0.9522 → 0.9048 (**−0.047**) | 0.848 → **0.580** |
| **P-gp** | 0.9587 → 0.9631 (**+0.004**) | 0.919 → 0.980 |

**The efflux axis transfers to unseen chemotypes; the permeability axis does not.** And the BBB
model's out-of-domain failure mode is specifically *calling almost everything permeant*
(specificity 0.580) — exactly the region where unmodelled efflux does its damage. The
specificity collapse is not an embarrassment to explain away; it is the mechanism of the harm.

**Do not say** the scaffold split "proves substantial generalizability." The table shows
specificity falling from 0.848 to 0.580 and MCC from 0.754 to 0.572. Examiners read tables.

### 7.2 Ablation (`results/ablation_study_results.csv`), 5-fold CV

| Task | Configuration | Features | ROC-AUC |
|---|---|:---:|:---:|
| P-gp | 3D shape only | 6 | 0.8425 ± 0.0243 |
| P-gp | 2D only | 6 | 0.9185 ± 0.0107 |
| P-gp | 2D + 3D | 12 | 0.9192 ± 0.0131 |
| P-gp | **Morgan + 2D** | 2,054 | **0.9509 ± 0.0070** |
| P-gp | Full (Morgan + 2D + 3D) — *deployed* | 2,060 | 0.9481 ± 0.0076 |
| BBB | 2D only | 6 | 0.9174 ± 0.0081 |
| BBB | Morgan only | 2,048 | 0.9393 ± 0.0054 |
| BBB | **Full hybrid** | 2,054 | **0.9467 ± 0.0035** |

Two readings, and give both:

1. **BBB synergy is real.** Fingerprints + descriptors (0.9467) beat either alone
   (0.9393, 0.9174). Topological substructure and global physicochemistry are complementary.
2. **The 3D descriptors do not earn their place.** 0.9481 *with* versus 0.9509 *without*. Six
   geometric variables alone reach 0.8425, confirming that shape genuinely correlates with
   cavity fit — but 2048-bit circular fingerprints already encode local stereochemical
   branching, so the information is redundant. **They are retained for mechanistic
   interpretability (Rg, asphericity in the SHAP story), not for accuracy.** This independently
   replicates Tiwari et al. (*Pharmaceutics* 2026) on the classification split.

> The project **title** claims a benefit "via 3D Conformational Modeling" that its own ablation
> table refutes. Know this before an examiner finds it. The correct position: the 3D arm is a
> published-standard negative replication plus an interpretability aid, and the title overclaims.

### 7.3 Cross-task SHAP (`results/shared_properties_quantitative_comparison.csv`)

| Descriptor | BBB mean \|SHAP\| | P-gp mean \|SHAP\| | Relationship |
|---|:---:|:---:|---|
| **TPSA** | **1.1825** | 0.1434 | Desolvation filter — BBB-specific |
| **MolLogP** | 0.3992 | **1.3949** | **Antagonistic** — helps influx, recruits efflux |
| MolWt | 0.5366 | 0.5783 | Compound barrier burden |
| HDonors | 0.2711 | 0.0911 | Seelig efflux trigger |
| HAcceptors | 0.1198 | 0.0708 | Polarity modifier |
| RotBonds | 0.1113 | 0.0387 | Induced-fit plasticity |

Ranking inverts completely:
- **BBB:** TPSA > MolWt > LogP > HDonors > RotBonds > HAcceptors
- **P-gp:** LogP > MolWt > Rg > NPR2 > TPSA > HDonors

Spearman ρ = 0.7714 across the six shared descriptors, **but p = 0.0724 at n = 6 — not
significant.** Report it as an observation, never as a result. Base values: φ₀ = −0.3121 (BBB),
+0.1519 (P-gp).

**The Golden CNS Window** that falls out of this: LogP ∈ [1.5, 3.0], TPSA < 70–90 Å²,
MW < 380–400 Da, HBD ≤ 1, Rg < 3.8 Å. Note it is an *intersection of opposing constraints* —
LogP must be high enough to enter and low enough to evade. That tension is the whole design
problem.

### 7.4 Clinical benchmarks (`core/decision_matrix.py`, Table 4)

| Drug | P(BBB+) | P(P-gp+) | Quadrant | Ground truth |
|---|:---:|:---:|---|---|
| Levetiracetam (Keppra) | 97.5% | 2.1% | **Q1** | BBB+ / P-gp− ✓ |
| Loperamide (Imodium) | 93.7% | 98.5% | **Q2** | BBB+ / P-gp+ ✓ |
| Caffeine | 88.0% | 6.8% | **Q1** | BBB+ / P-gp− ✓ |
| Diazepam (Valium) | 99.4% | 6.2% | **Q1** | BBB+ / P-gp− ✓ |
| Penicillin G | 1.3% | 3.4% | **Q3** | BBB− / P-gp− ✓ |

**Volunteer this immediately: these are recall checks, not blind validation.** By InChIKey
match, all five are in B3DB and three (Loperamide, Caffeine, Diazepam) are also in Broccatelli.
Only Levetiracetam and Penicillin G are unseen by the P-gp model. The table shows the model
reproduces established pharmacology; the **scaffold split is the only generalisation evidence
claimed.**

#### Illustration molecules — and their strict limit

**Read this first: single molecules are illustration, never evidence.** n=1 proves nothing, and
hand-picking molecules whose pharmacology you already know is cherry-picking. **The aggregate
numbers are the evidence, and you already have them:** the scaffold split (1,171 and 241
molecules, zero scaffold overlap with training) *is* the out-of-sample experiment. A molecule's
only job is to make the concept land in a listener's head in the first 60 seconds. Always say
the aggregate number in the same breath.

Loperamide explains the *concept* better than anything else, so keep it for that. But it is in
both training sets, so **never present it as a prediction.** Checked at Bemis-Murcko
connectivity-block level (stricter than full InChIKey, so stereoisomer and salt-form variants
cannot sneak through), exactly three showcase drugs are absent from *both* benchmarks:

| Drug | P(BBB+) | P(P-gp+) | Quadrant | Independent corroboration |
|---|---:|---:|:---:|---|
| **Linezolid** | 0.967 | 0.017 | **Q1** | Clinical CSF/plasma ratio **> 0.70** — a fact from pharmacokinetics, not from any training label |
| **Lurasidone** | 0.950 | 0.968 | **Q2** | **Tightest binder of all 22 docked compounds, −10.20 kcal/mol, Kd 33.4 nM** |
| **Pimavanserin** | 0.982 | 0.961 | **Q2** | −8.87 kcal/mol, Kd 313 nM |

Lurasidone is the cleanest *illustration*: unseen by both training sets, called Q2 by the
ligand-topology model, and independently the tightest P-gp cavity binder in the structural
study. But it is **one point in a 22-compound scatter**, and the scatter — not the point — is
the result: Welch p = 1.6×10⁻³, Spearman ρ = −0.5733. Quote the scatter; use the point only to
make it concrete.

Drugs that look out-of-sample but are **not** (their skeletons are in B3DB, caught only by the
strict check): Naloxone, Chloramphenicol, Betamethasone, Triamcinolone, Milnacipran, Everolimus.
Do not call any of these blind predictions.

> **If asked "isn't that cherry-picking?" — say yes.** "Correct, and that's why it's an
> illustration rather than a result. Three drugs chosen because their pharmacology is
> independently documented is a biased sample. The systematic version is the scaffold split:
> 1,171 BBB and 241 P-gp molecules with zero scaffold overlap against training, ROC-AUC 0.9048
> and 0.9631 with bootstrap intervals. That's the number I'd defend."

#### Every anecdote has an aggregate counterpart — lead with the right column

| Illustration (n=1) | The evidence to actually quote |
|---|---|
| Lurasidone: ML and docking agree, unseen drug | 22 compounds: Q2 −8.84±0.89 vs Q1 −6.91±1.30, Welch p = 1.6×10⁻³, ρ = −0.5733 |
| Linezolid: CSF/plasma > 0.70 corroborates | Scaffold split: BBB 0.9048 [0.8835–0.9240], P-gp 0.9631 [0.9359–0.9846] |
| Loperamide is paradoxical | 45/140 = **32.1%** of BBB+ are substrates [24.5, 40.6]; 240 Q2 drugs pharmacopeia-wide |
| Everolimus: brain-tumour drug, predicted effluxed | **105** marketed CNS drugs in Q2 |

Worked SHAP, Loperamide: margin +2.6929 (BBB) and +4.2183 (P-gp), each satisfying
`P = 1/(1+e^{−f})`. φ_LogP = +0.2074 (BBB) but **+2.2500** (P-gp) at LogP 5.09; φ_Rg = +0.4227
at Rg = 5.07 Å. That is the paradox, decomposed into numbers: the same lipophilicity that gets
it in is what gets it pumped out.

### 7.5 Pharmacopeia screen (2,405 approved drugs)

| Quadrant | Meaning | Count |
|---|---|---|
| **Q1** | BBB+ / P-gp− — reaches the brain | **710 (29.5%)** |
| **Q2** | BBB+ / P-gp+ — **paradoxical** | **240 (10.0%)** |
| **Q3** | BBB− / P-gp− — peripheral | **1,064 (44.2%)** |
| **Q4** | BBB− / P-gp+ — doubly restricted | **391 (16.3%)** |

- **273 non-CNS approved drugs in Q1** → repurposing leads. Top hits: Thalidomide
  (S_CNS 0.9534, glioblastoma), Betamethasone (0.9821, vasogenic oedema), Aminoglutethimide
  (0.9632, brain metastases), Linezolid (0.9510, resistant meningitis — and its known
  CSF/plasma ratio > 0.70 is independent corroboration), Milnacipran (0.9835, central
  neuropathic pain), Topotecan (0.6237, SCLC brain metastases).
- **105 marketed neuropsychiatric drugs in Q2** → a structural rationale for treatment
  resistance: Flupentixol (P(P-gp+) 0.988), Zuclopenthixol (0.985), Vilazodone (0.982),
  Everolimus (0.976, prescribed for SEGA — a brain tumour), Lurasidone (0.968).
- Disease-area pattern: oncology and cardiology drugs sit mostly in Q3/Q4; neuropsychiatric
  drugs are **bifurcated** — successful agents cluster in Q1, second-line and
  treatment-resistant agents slip into Q2.

### 7.6 Cryo-EM docking (22 compounds, PDB 6QEX)

| Group | n | Mean ΔG_bind (kcal/mol) |
|---|:---:|:---:|
| **Q2 efflux-vulnerable** | 9 | **−8.84 ± 0.89** |
| **Q1 efflux evaders** | 10 | **−6.91 ± 1.30** |
| Q3 peripheral | 2 | — |
| Q4 restricted (Paclitaxel) | 1 | −8.88 |

- Welch's t = **−3.7933**, p = **1.597 × 10⁻³**
- Mann-Whitney U = 11.0, p = 3.117 × 10⁻³
- Spearman ρ(P(P-gp+), ΔG) = **−0.5733**, p = 5.28 × 10⁻³

Tightest: Lurasidone −10.20 (Kd 33.4 nM), Vilazodone −9.95 (50.6 nM), Loperamide −9.39
(131.4 nM). Weakest: Metformin −4.94, Caffeine −5.55, Levetiracetam −5.59 (Kd ~80 μM — no
meaningful interaction). Contact network engaged by Q2 drugs: **Trp232, Phe303, Ile306, Tyr307,
Tyr310, Phe336, Leu339, Ile340, Phe343, Gln725** on TM4/5/6/12.

Two compounds landed where the hand-assignment did not expect, and are reported as the model
places them: **Paclitaxel → Q4** (predicted BBB-negative) and **Aspirin → Q1** (predicted
BBB-positive). Reporting these rather than suppressing them is the point of deriving quadrants
from live predictions.

> **The limitation you must state yourself, because it is also the argument for your method:**
> the group separation is real, but it **does not transfer to individual compounds.** Four Q1
> evaders (Thalidomide −8.23, Topotecan −8.68, Linezolid −7.62, Betamethasone −8.71) bind at
> least as tightly as the weakest Q2 substrates, and the peripheral control **Penicillin G
> reaches −8.96** despite a predicted efflux probability of 0.020 — tighter than six of the
> nine Q2 substrates. Rigid-receptor ΔG is a **population-level signal, not a per-molecule
> discriminator.** This matches published reports that P-gp docking cannot separate substrates
> from non-substrates — **and it is precisely why a ligand-topology model adds information the
> structure alone does not.**

---

## 8. The multi-task GNN

### 8.1 Why it was tried, and what was declared in advance

**Start here, every time this topic comes up.** The outcome was pre-registered before any run,
Quoted from the pre-run project plan (`git show 8af9c2f^:docs/RESEARCH_LOG.md`, line 1021):

> *"if the P-gp head does not improve, the headline shifts to C2+C3+C4 — the measurement, the
> evaluation, and the attribution — with multi-task reported as a negative result."*

So both branches were declared in writing in advance, and the hypothesis itself was asymmetric:
**P-gp should gain, BBB might not.**

Why it was worth testing. A 2025 review proposed exactly this as future work: *"multi-task
learning leverages auxiliary signals such as transporter affinity to guide extraction of
disentangled, mechanism-relevant features."* And the precondition looked satisfied — the two
labels are near-independent (Cramér's V = 0.064), and near-independent labels are
*complementary* rather than redundant supervision, which is the textbook condition for
multi-task learning to help the smaller task. P-gp has 848 training molecules against BBB's
5,463, so P-gp was the task expected to gain.

**Why the resulting null is a contribution and not a failure is argued in full in §2** — the
short form being that the independence measurement *predicts* this outcome, so a positive
result would have contradicted the project's own lead claim.

### 8.2 Architecture (`gnn/`)

```
SMILES → RDKit graph:  x[V, 66]   edge_index[2, 2E]   edge_attr[2E, 14]
             │
   3× [ GINEConv(edge_dim=14) → BatchNorm1d(256) → ReLU → Dropout(0.30) ]   ← shared encoder
             │
   global_mean_pool  →  h[B, 256]                                           ← heads branch HERE
             │
      ┌──────┴──────┐
      ▼             ▼
  head "bbb"    head "pgp"        each: Linear(256→32) → ReLU → Dropout → Linear(32→2)
```

**Feature encoding** (`featurizer.py`), two modes:

- `legacy` — the base paper's 9 atom / 7 bond raw ordinals. **Retained only for parity and
  reproducibility, not recommended:** atomic number enters as a raw magnitude so chlorine (17)
  looks like "17× carbon" (6), hybridisation enters as a raw enum ordinal, nothing is
  normalised, and bond features 4 and 5 are near-collinear (`bond_type == AROMATIC` versus
  `bond.GetIsAromatic()`).
- `encoded` — **66 atom features**: one-hots for element (22+1), degree (6+1), formal charge
  (5+1), numHs (5+1), hybridisation (6+1), chirality (4+1), smallest ring size (5+1), each with
  a trailing catch-all slot so an unseen value cannot collide with a real one; plus 3 binary
  flags and 3 **bounded** continuous features (valences /8, mass /100 — fixed constants, not
  data statistics, so no leakage path exists). **14 bond features**: bond type (4+1), stereo
  (6+1), conjugated, in-ring; the redundant aromatic flag dropped.

Dimensions are computed from a probe molecule by `feature_dims()`, never hand-written — the
base paper hardcoded 9 and 7 as literals in both the model and the checkpoint metadata, which is
how metadata starts silently lying after a refactor. Checkpoints here carry
`architecture_metadata()` read off the live module, and `build_model_from_checkpoint()`
reconstructs from it.

**Parameter cost:** 345,514 (legacy, 1 head — matches the reference implementation) → 364,608
(encoded, 1 head) → **372,898 (encoded, 2 heads)**. The second barrier costs **+8,290
parameters, ~2.3%**. Branching after the 32-unit `fc1` instead would cost only +66 parameters —
but would force both tasks through a 32-unit shared bottleneck. Branching at the pooled
256-dim vector is the deliberate choice.

### 8.3 Masked loss (`losses.py`) — why it is unavoidable

~216 of ~8,801 molecules carry both labels, so **~97.5% have exactly one.** A plain summed
cross-entropy over two heads would train each head on fabricated labels for the molecules it
has no ground truth for. Instead:

- each head's loss is computed over **only its masked-in molecules**;
- `reduction='mean'` normalises by **that head's own labelled count**, so the 7,805-molecule
  BBB task cannot swamp the 1,212-molecule P-gp task by sheer weight of numbers;
- a head with no labelled molecules in a batch receives **exactly zero gradient** — asserted in
  the verification suite;
- per-task inverse-frequency class weights are computed from the **train partition only** —
  computing them over the whole corpus would leak test class balance.

### 8.4 Splits (`splits.py`) — the invariant that matters

**Partitioning happens at the level of the MOLECULE, never the task.** A dual-labelled molecule
must carry *both* labels into the same partition; if a molecule were in train for BBB and test
for P-gp, every multi-task number would be contaminated. `assert_split_integrity()` is called
before training and checks, in order of how badly each would invalidate the work:

1. no InChIKey in more than one partition;
2. every molecule assigned exactly once;
3. every dual-labelled molecule keeps both labels in one partition;
4. for a scaffold split, **zero** scaffold overlap between train and test;
5. each partition has both classes for each task it carries (otherwise ROC-AUC is undefined).

`random` stratifies on the composite **(label, mask) pattern** — e.g. `'1_x'` is a BBB-positive
molecule with no P-gp label — so both label distributions *and* the single/dual-label mix are
preserved; rare strata merge into one bucket so sklearn cannot fail on a class of size 1.
`scaffold` follows the DeepChem convention (largest clusters fill **train** first), deliberately
leaving rare chemotypes in val/test — the honest hard case.

### 8.5 Result: it does not work, and that is the finding

**Random split**

| Task | Experiment | n | ROC-AUC | 95% CI | MCC | F1 |
|---|---|---:|---:|---|---:|---:|
| bbb | single_bbb | 1,172 | 0.9388 | [0.9246, 0.9531] | 0.7264 | 0.9012 |
| bbb | multitask | 1,172 | 0.9404 | [0.9270, 0.9533] | 0.7078 | 0.8915 |
| pgp | single_pgp | 182 | 0.9390 | [0.8964, 0.9728] | 0.7460 | 0.8821 |
| pgp | multitask | 182 | 0.9301 | [0.8810, 0.9688] | 0.7460 | 0.8821 |

**Scaffold split**

| Task | Experiment | n | ROC-AUC | MCC |
|---|---|---:|---:|---:|
| bbb | single_bbb | 1,036 | 0.7583 | 0.3927 |
| bbb | multitask | 1,036 | 0.7467 | 0.2879 |
| pgp | single_pgp | 333 | 0.8227 | 0.5442 |
| pgp | multitask | 333 | 0.8171 | 0.5389 |

All four confidence intervals overlap. **Multi-task training is neutral to mildly harmful on
both tasks, on both splits. Neither half of the pre-registered hypothesis holds.**

Three pieces of evidence establish this is a real negative result and not a bug:

1. **The plumbing is verified.** `sanity_pgpw0` — the multi-task architecture with the P-gp
   loss weight set to 0 — gives BBB ROC-AUC **0.9411** against the single-task 0.9388 (within
   noise), while its untrained P-gp head sits at **0.5626, i.e. chance.** Weight 0 genuinely
   removes a task, and the trunk trains exactly as it does single-task.
2. **The obvious confound is ruled out.** Thin P-gp gradient signal was the likeliest innocent
   explanation. Upweighting P-gp does **not** rescue it — ROC-AUC 0.9295 at weight 3, 0.9299 at
   weight 5, versus 0.9390 single-task — and it progressively **damages BBB**: 0.9388 → 0.9310
   → 0.9086, MCC 0.7264 → 0.6821 → 0.6467. **The tasks compete for encoder capacity instead of
   helping each other.**
3. **It coheres with the label measurement.** Multi-task learning borrows strength from
   *correlated* auxiliary signal. At Cramér's V = 0.064 there is none to lend. **The negative
   result is what the independence measurement predicts.** The two findings are one story.

**Chemotype transfer, reproduced in a second model family:**

| Experiment | Task | random → scaffold |
|---|---|---|
| single_bbb | bbb | 0.9388 → 0.7583 (**−0.1805**) |
| single_pgp | pgp | 0.9390 → 0.8227 (**−0.1163**) |

BBB degrades ~55% more than P-gp under chemotype shift. The XGBoost arm showed the same
asymmetry. A finding that survives a change of model family is much harder to dismiss as an
artifact.

### 8.6 Seed variance — why one run is not evidence (`run_seed_variance.py`)

The measured effects are of order 1e-3. A single run cannot distinguish that from run-to-run
noise, so the null it implies is **not yet evidence**. Five seeds, **paired** (the same seed
drives both the split and the initialisation, for every experiment):

| | BBB | P-gp |
|---|---|---|
| multi-task mean ROC-AUC | 0.9313 | 0.9033 |
| single-task mean | 0.9407 | 0.9060 |
| **delta** | **−0.00944** | **−0.00272** |
| delta 95% CI | [−0.02337, +0.00449] | [−0.03479, +0.02936] |
| sd of differences | 0.01122 | 0.02583 |
| **MDE at 5 seeds, 80% power** | **0.01865** | **0.04295** |
| \|delta\| below MDE | **yes** | **yes** |
| paired t-test | t = −1.881, p = 0.1331 | t = −0.235, p = 0.8256 |
| **P(multi > single)** | **0.200** | 0.540 |
| seeds needed for 80% power | **14** | **712** |

**What this licenses, stated precisely:** neither delta is resolvable at 5 seeds, so the correct
conclusion is *"no effect detected at this sample size"* — **never "no effect exists."** But the
two tasks are not in the same position:

- **BBB:** P(multi > single) = 0.200 — single-task wins four cross-seed pairings in five. By
  Bouthillier et al.'s own 0.75/0.25 criterion that is a reliable *ordering*, even though the
  paired t-test is non-significant. The evidence leans toward multi-task being mildly harmful
  to BBB, and **14 seeds would settle it.**
- **P-gp:** P(multi > single) = 0.540, indistinguishable from a coin flip; resolving a −0.0027
  delta would need **712 seeds**. An effect that small is practically zero whatever its sign.
  This is the strongest form the P-gp null can take.

The MDE-over-seeds methodology is **adopted and cited, not claimed** — a 2026 PROTAC-activity
preprint applies the same reasoning and reports a +0.0025 AUROC effect as "informative as a
null result."

### 8.7 K-fold, because single splits leave too few dual labels

A single split leaves only 33 (random) or 48 (scaffold) dual-labelled molecules in test — too
few to score quadrant decisions on. `run_kfold_multitask.py` runs 5 stratified folds over the
whole corpus so **every labelled molecule gets one honest out-of-fold prediction**, including
all 216 dual-labelled ones. A validation slice is carved from each fold's *training* portion, so
the held-out fold is never seen during training **or model selection**. `assert_oof_integrity()`
then checks: no molecule predicted twice, every labelled molecule covered, and **no molecule
scored by a fold that trained on it.**

---

## 9. Decision-aware metrics

### 9.1 Why they exist

Standard ADMET evaluation scores each endpoint separately. A model can report ROC-AUC 0.94 on
BBB and 0.94 on P-gp while placing a large fraction of molecules in the **wrong therapeutic
quadrant**. Nothing in the literature evaluates the dual-barrier decision jointly. The
mathematics is standard; the contribution is the framing — these are the numbers that matter for
the CNS progression decision, and none of them is currently reported.

| Metric | Definition |
|---|---|
| **M1 quadrant exact match** | Both axes correct (subset accuracy over 4 classes) |
| **M2 quadrant κ** | Chance-corrected 4-class agreement, plus the full 4×4 confusion matrix |
| **M3 paradox recall / precision / F1 / leakage** | Q2 as the positive class. **Leakage rate** = true-Q2 molecules the model calls Q1 — the single most consequential error in CNS triage |
| **M4 efflux-disparity gap** | `AUC_BBB(non-substrates) − AUC_BBB(substrates)` — is permeability prediction systematically worse where efflux is present? |
| **M5 decision regret** | Expected cost of the progress/don't-progress decision as a **curve** in λ = c_FP / v_hit: `regret(λ) = (λ·n_false_progress + n_missed) / n`. Reporting a curve avoids inventing dollar values, which is what makes it defensible |

Three strategies scored on all 216 dual-labelled molecules:
**A** BBB-only (the published status quo — scoring it means assuming P-gp-negative throughout),
**B** composed two independent single-task models, **C** the shared-encoder multi-task model.

### 9.2 M1–M3

| Evaluation | Exact match | κ | Paradox recall | Precision | **Leakage** |
|---|---:|---:|---:|---:|---:|
| **A** BBB-only | 0.5556 [0.4769, 0.6204] | 0.2742 | **0.0000** | 0.0000 | **0.9333** |
| **B** Composed | **0.5787** [0.5093, 0.6481] | **0.3725** | 0.4444 | 0.4348 | 0.4889 |
| **C** Multi-task | 0.5602 [0.4954, 0.6251] | 0.3478 | **0.5111** | **0.4792** | **0.4222** |

**The headline: per-axis ROC-AUC is ~0.94, yet only 56–58% of molecules land in the correct
quadrant. Four in ten joint decisions are wrong, and no currently reported metric shows this.**

**Paradox leakage** — true Q2 molecules progressed as Q1: **42 of 45 (93.3%)** for a single-axis
model, 22/45 (48.9%) composed, 19/45 (42.2%) multi-task. A single-axis model advances 93% of
paradoxical compounds as though they reach the brain. A second axis roughly halves that. B and C
are indistinguishable.

*Be fair about A:* a single-axis BBB model scores exactly 0 on paradox recall **by
construction**. That is a statement about what single-axis evaluation can express, not a fair
contest between models.

### 9.3 M4 — the hypothesis was wrong, and the mechanism explains why

The prediction was that BBB performance would be *worse* on efflux substrates. It is
significantly **better**:

| Evaluation | AUC non-substrate (n=152) | AUC substrate (n=64) | Gap | Verdict |
|---|---:|---:|---:|---|
| A / B | 0.7990 | 0.9368 | **−0.1379** [−0.2406, −0.0410] | gap detected |
| C | 0.7904 | 0.8760 | −0.0856 [−0.1992, +0.0247] | CI spans zero |

Why — measured, not assumed. The two strata differ sharply in chemistry:

| | MolWt | LogP | TPSA | HDonors | RotBonds |
|---|---:|---:|---:|---:|---:|
| non-substrate | 281.4 | 2.2 | 60.3 | 1.4 | 3.9 |
| substrate | 370.9 | 3.8 | 47.8 | 0.8 | 5.2 |

And permeability is *more* separable among substrates — the BBB+ vs BBB− TPSA gap is **−28.9**
there versus **−21.5** among non-substrates. P-gp substrates are a chemically narrow, drug-like
region where polarity cleanly determines permeability; non-substrates are a heterogeneous mix
including small polar molecules where permeability is genuinely harder to call.

**This sharpens the argument rather than weakening it.** The harm from unmodelled efflux is not
that permeability prediction degrades — it is that permeability prediction is **accurate, and
that accurate prediction is clinically misleading.** The molecule really is permeant and really
is pumped back out. The 93.3% leakage is the harm; there is no accuracy deficit to point at.
Reporting a falsified prediction with its mechanism is stronger than quietly dropping it.

### 9.4 M5 — the ranking depends on the cost ratio

| Evaluation | λ=0.25 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A BBB-only | **0.1296** | **0.2083** | 0.3657 | 0.6806 | 1.3102 | 2.5694 | 5.0880 |
| B Composed | 0.1806 | 0.2268 | **0.3194** | **0.5046** | **0.8750** | **1.6157** | **3.0972** |
| C Multi-task | 0.1921 | 0.2407 | 0.3380 | 0.5324 | 0.9213 | 1.6991 | 3.2546 |

Regret is linear in λ, so the crossover solves exactly: **λ = 0.643** against B, **λ = 0.769**
against C. **Below that, the single-axis model is genuinely the better decision rule** — it
progresses more compounds and incurs fewer missed opportunities (11 versus 29 and 31). The
efflux axis only pays once a false progression costs more than roughly 0.7× a missed
opportunity. In real CNS development, where a failed programme costs far more than a passed-over
lead, λ ≫ 1 — but say so as a judgement, not a measurement.

**That nuance is the entire argument for reporting a curve.** A single headline number would
have hidden it.

**Honest limits of M1–M5, which belong in any write-up that uses them:** dual labels exist for
only ~216 molecules, so every interval is wide; Q4 has ~19 molecules so its conditional numbers
are indicative only; and strategy A scores 0 on paradox recall by construction.

---

## 10. Weaknesses and honest answers

Rehearse these. Volunteering a limitation before it is extracted from you converts a weakness
into evidence of rigour.

| # | Weakness | The honest answer |
|---|---|---|
| 1 | **"A dual-barrier CNS map was published in 2004."** Adenot & Lahana, *J. Chem. Inf. Comput. Sci.* 2004, 44, 239–248 built a BBB model *and* a P-gp model and proposed a combined CNS map. **The PDF is in this repo**, at `Initial Model/reference_papers/adenot_2004_jcim.pdf`. | **Concede immediately and reframe.** "Building a dual-barrier tool is not our claim — that is prior art, and it is cited. Our claim is the *measurement*: 32.1% of B3DB BBB+ molecules are confirmed P-gp substrates, the labels carry no usable mutual information, and consequently 93.3% of paradoxical compounds leak through single-axis evaluation. Adenot built the tool; we quantified what the field loses by not using one." |
| 2 | **The 3D descriptors in the title add nothing** (0.9481 with vs 0.9509 without). | State it first. It is an independent replication of Tiwari et al. 2026 on the classification split. The 3D block is retained for mechanistic interpretability, not accuracy, and **the title overclaims — it should be revised.** |
| 3 | **ROC-AUC 0.9522 is mid-pack.** Published B3DB results cluster at 0.93–0.95; a 2025 geometric GNN reports 0.9685 *under scaffold splits*. | "This is not a performance paper. The models are competent instruments for the measurement, not the contribution. If it were a leaderboard paper it would not be competitive, which is exactly why the claim is a diagnostic one." |
| 4 | **The clinical benchmark drugs are in the training sets.** | Volunteer it. Table 4 is explicitly labelled a recall check. A prior blind-validation claim was **retracted in writing** after the Loperamide SMILES was found to have one extra CH₂ (it was the *wrong structure* that was absent from both databases). The scaffold split is the only generalisation evidence claimed. |
| 5 | **The scaffold comparison is not apples-to-apples** — `run_validation.py` trains a fresh, *untuned* model for that protocol. | Say so unprompted. The direction and size of the effect are unambiguous, but the magnitude partly conflates protocol with tuning. The fix is to re-tune inside the scaffold protocol; it is listed as future work. |
| 6 | **"Your scaffold split proves generalisability" — no, your own table shows specificity 0.848 → 0.580.** | Never claim it does. The collapse **is** the finding: out of domain, the BBB model calls almost everything permeant, which is exactly where unmodelled efflux does damage. |
| 7 | **Only 216 dual-labelled molecules.** Every joint number rests on them. | Stated in the limitations of README, the manuscript and `decision_metrics.py` itself. The OR interval reaches 2.67; Q4 has ~19 molecules. The prevalence claim (32.1%, Clopper-Pearson [24.5, 40.6]) is the robust one because it is a proportion, not a hypothesis test. |
| 8 | **The multi-task GNN failed. Isn't that a negative result?** | Yes, and it is reported as one — **pre-registered** (the pre-run plan declares both branches; `git show 8af9c2f^:docs/RESEARCH_LOG.md`), with a verified sanity control (weight-0 removes a task; the untrained head sits at 0.5626 = chance), the obvious confound ruled out (upweighting does not rescue and actively damages BBB), and a mechanistic explanation (V = 0.064 means no correlated signal to borrow). A 2025 review proposed this architecture as future work; it is now tested on both standard public benchmarks and found not to deliver. **Had it worked, it would have contradicted the project's own central measurement — see §2 for the full argument and the hostile-follow-up drill.** |
| 9 | **Five seeds is not many.** | Agreed, and that is why the MDE is reported. Both deltas fall below the MDE, so the conclusion is "no effect detected at this sample size", never "no effect exists". 14 seeds would settle BBB; P-gp would need 712, which is itself the finding. |
| 10 | **Docking is rigid-receptor at 3.5 Å, exhaustiveness 8 — isn't that weak?** | Yes, and it is reported as weak at the per-molecule level. 3.5 Å does not resolve waters, P-gp's induced fit is unmodelled, and Vina's scoring function is empirical. Penicillin G at −8.96 kcal/mol with 2% predicted efflux is the proof: group separation is significant (p = 1.6e-3), individual discrimination is not. **That is the argument for the ML model, not against it.** |
| 11 | **`pgp_model_training.py` passes the test set in `eval_set`.** | True, and it is monitoring only — there is **no `early_stopping_rounds`** on that model, so no model selection depends on the test set. It would be cleaner to drop it; flagged. The BBB model, which *does* early-stop, stops on the **validation** split. |
| 12 | **The two arms use opposite scaffold-split conventions.** `run_validation.py` fills **test** from the largest clusters; `gnn/splits.py` fills **train** first (DeepChem convention). | A real inconsistency. The GNN convention is the defensible one — it leaves rare chemotypes in test, the honest hard case, whereas filling test from the largest clusters gives a chemically narrow test set. The XGBoost arm should be aligned. |
| 13 | **The screen's applicability domain uses the whole dataset as reference, not just the training split.** | Correct, so the reported T_max is mildly optimistic. It is used as a coarse novelty filter (threshold 0.35) rather than a performance claim, so the effect on conclusions is small — but it should be training-only. |
| 14 | **`counterfactual.py` re-implements featurisation instead of calling `core.features`.** | A genuine drift risk — the kind of duplication that caused the base paper's two featurisers to diverge on the valence API. Behaviour currently matches; it should be refactored to the shared path. |
| 15 | **`docs/ARCHITECTURE.md` still quotes superseded docking numbers** (ΔG −9.23 ± 0.69, p = 0.0035, t = −4.10) and its §8 layout lists a `pipelines/` directory that does not exist. | Known documentation drift. The live numbers are the manuscript's and `results/cryoem_pgp_docking_validation.csv`'s: −8.84 ± 0.89 vs −6.91 ± 1.30, t = −3.7933, p = 1.6e-3. `verify_doc_consistency.py` enforces the README and manuscript; ARCHITECTURE.md's §2.5 and §5.4 are not yet under that check. **Fix before submission.** |
| 16 | **The manuscript's `U_dual` uncertainty formula does not match the code.** The manuscript gives `1 − 2·|P_BBB − 0.5|·|P_Pgp − 0.5|`; `inference.py` reports `max` of the two per-model scores `1 − |p − 0.5|/0.5`. | The code is self-consistent and documented in `applicability_domain.py`; the manuscript formula is stale. Fix the manuscript to match the code. |
| 17 | **Predicting BBB+ / P-gp− does not guarantee efficacy.** | Stated as limitation 1 of the manuscript. Net exposure also depends on unbound brain fraction, plasma protein binding, BCRP and MRP efflux, and metabolic clearance. BCRP is the named next model to add. |
| 18 | **In vitro P-gp labels are noisy** (calcein-AM vs bidirectional Caco-2/MDCK, lab-to-lab variability). | Acknowledged. It is why every metric carries a bootstrap CI and why the applicability domain is reported alongside each prediction. |

---

## 11. Likely viva questions

**Q: Explain your project in one sentence.**
We model the two barriers that decide whether a drug reaches the brain — passive BBB permeation
and active P-glycoprotein efflux — and we measure what the standard benchmark loses by modelling
only the first.

**Q: Why not just one model with three classes?**
Because the two barriers are *mechanistically* independent processes with different physics, and
we measured that their labels are statistically near-independent (Cramér's V = 0.064). Collapsing
them into one label discards the decoupling that is the entire point, and it would make the
paradoxical class (Q2) inexpressible as a *combination* rather than a category.

**Q: Why XGBoost rather than a neural network?**
The feature space is a 2,048-bit sparse binary fingerprint plus six continuous descriptors on
7,805 molecules. Gradient-boosted trees are the right inductive bias for sparse tabular data at
that scale, they train in seconds, and crucially **TreeSHAP gives exact Shapley values** on
them — the explainability layer is a reason for the model choice, not an afterthought. We *did*
build the GNN arm, and on the same data it reaches 0.9388 ROC-AUC versus XGBoost's 0.9522.

**Q: What is the difference between a random split and a scaffold split, and which do you
believe?**
A random split can place a near-twin of every test molecule in train, so it measures
interpolation. A scaffold split guarantees no shared Bemis-Murcko core, so it measures
generalisation to new chemistry. **The scaffold number is the honest one.** On BBB it costs
0.047 ROC-AUC and 0.27 specificity; on P-gp it costs nothing. That asymmetry is a result.

**Q: Why log-odds margin space for SHAP?**
Because the additive decomposition `f(x) = φ₀ + Σφᵢ` holds *exactly* only in margin space. The
sigmoid is non-linear, so attributions computed on probabilities do not sum to the prediction.
We report the margin, verify `P = 1/(1+e^{−f})` against the stated probability, and recover
probability via the logistic link.

**Q: What is φ₀ and what does a negative value mean?**
The expected margin over the training background: −0.3121 for BBB, +0.1519 for P-gp. A negative
base means that, absent any feature information, the BBB model leans slightly toward
non-permeant. Every attribution is a deviation from this baseline.

**Q: A fingerprint bit has high attribution. So what?**
On its own, nothing — that is why `MorganSubstructureMapper` exists. It reads RDKit's `bitInfo`
to recover the (centre atom, radius) that set the bit, rebuilds the fragment with
`FindAtomEnvironmentOfRadiusN` + `PathToSubmol`, and returns a fragment SMILES plus a
functional-group tag. Caveat: at 2048 bits collisions occur, so single-bit claims are weaker
than descriptor-level ones.

**Q: How do you know a prediction is trustworthy?**
Two independent checks plus one confidence measure. **Max Tanimoto** against the reference
library — has the model seen anything structurally like this? **Descriptor leverage**,
`sqrt(mean(z²))` — are the physical properties inside the training range? Inside domain needs
both (`T ≥ 0.40`, leverage ≤ 2.5). Separately, an **uncertainty score** `1 − |p − 0.5|/0.5` that
is 1.0 at the decision boundary. A molecule can be inside the domain and still uncertain, or
outside the domain and spuriously confident — which is the case the domain check exists to catch.

**Q: Why ETKDGv3 with a fixed seed?**
Conformer generation is stochastic. Without `randomSeed = 42` the 3D descriptors — and therefore
the P-gp prediction — would change between runs, and no reported number would be reproducible.
Same reasoning behind `--seed 42` for Vina, which seeds from the clock by default.

**Q: Your counterfactuals — why reaction SMARTS and not gradient-based perturbation?**
Because perturbing a feature vector produces molecules that do not exist. We apply eight real
medicinal-chemistry transformations, sanitise the product in RDKit, require **Tanimoto ≥ 0.50**
to the parent so the analogue is recognisably related, re-generate a fresh 3D conformer, and
re-predict. Every suggestion is a molecule a chemist could actually make.

**Q: Why S_CNS = P(BBB+) × (1 − P(P-gp+))?**
It is the probability of the conjunction "permeant **and** not effluxed" under an independence
assumption — and we measured that assumption holds well enough (V = 0.064) for it to be
reasonable. It is monotone in the right direction on both axes and collapses to ~0 if either
fails, which is the clinical behaviour. It is a ranking score, not a calibrated probability of
brain exposure, and should be presented as such.

**Q: Why does the docking validate anything if it cannot classify individual compounds?**
It validates at the level it is powered for. The ML model is trained on *phenotypic cell assays*
and knows nothing about the protein structure; the docking knows only the structure and nothing
about the assay. They agree on a group separation at p = 1.6e-3 and a rank correlation of
ρ = −0.5733. That is two independent modalities converging. The per-molecule failure
(Penicillin G at −8.96 kcal/mol with 2% predicted efflux) is reported, and it is the argument
for the ligand-topology model — the structure alone does not contain the answer.

**Q: Why did you include the Paclitaxel that is already in the structure?**
As a positive control on the docking box. It is the co-crystallised ligand, so it should bind
tightly in the cavity we defined — it does, at −8.88 kcal/mol. Notably our own model predicts it
BBB-negative, so it derives to **Q4**, not the Q2 it was hand-assigned. We report it as the model
places it; that disagreement is printed at runtime rather than suppressed.

**Q: Why was the multi-task GNN expected to work, and why didn't it?**
Expected because the two labels are near-independent, and near-independent labels are
complementary rather than redundant supervision — the textbook precondition for the smaller task
(848 P-gp training molecules versus 5,463 BBB) to gain. It failed because that same independence
means there is no *correlated* signal to borrow. Multi-task learning transfers shared structure;
V = 0.064 says there is almost none. Instead the two tasks compete for encoder capacity, which
is why upweighting P-gp damages BBB (0.9388 → 0.9086) without helping P-gp.

**Q: How do you know the multi-task failure isn't a bug in your code?**
Three controls. (1) **Weight-0 sanity run:** the multi-task architecture with P-gp weight 0
reproduces single-task BBB (0.9411 vs 0.9388) while its untrained P-gp head sits at 0.5626 —
chance. So the masking genuinely zeroes a task and the trunk trains normally. (2) **29/29
verification checks**, including an assertion that a head with no labelled molecules in a batch
receives exactly zero gradient. (3) The failure is **consistent across both splits, both tasks,
and three task-weight settings** — bugs are rarely that orderly.

**Q: Why is the masked loss normalised per head rather than globally?**
Because `reduction='mean'` over the pooled batch would weight each task by its molecule count,
and BBB has roughly 6.4× more labelled molecules. The P-gp head would be a rounding error in
the gradient. Normalising by each head's own mask count makes the task weighting an explicit,
tunable choice rather than an accident of dataset size.

**Q: Why must splits be at molecule level, not task level?**
If a dual-labelled molecule were in train for BBB and test for P-gp, the encoder would have seen
that exact molecule's structure during training and every multi-task number would be
contaminated. `assert_split_integrity()` enforces it before training and raises rather than
warning.

**Q: What is MDE and why does it matter here?**
Minimum Detectable Effect — the smallest true difference this design could detect at 80% power.
At 5 seeds it is 0.01865 for BBB and 0.04295 for P-gp. Both measured deltas (−0.0094, −0.0027)
fall below it. So the licensed conclusion is "no effect detected at this sample size", not "no
effect exists". Stating it the second way would be a claim the data cannot support.

**Q: Your per-axis AUC is 0.94 but quadrant accuracy is 56%. Isn't that a contradiction?**
No — it is the point. Joint accuracy multiplies two independent error channels, and both axes
must be right simultaneously. 0.94 AUC still leaves ~11% error per axis at threshold, and the
errors do not cancel. **No metric currently reported in the literature exposes this**, which is
why M1–M5 exist.

**Q: What is paradox leakage and why is it the number you lead with?**
True Q2 molecules — permeant but effluxed — that the model advances as Q1. A single-axis BBB
model leaks **42 of 45, or 93.3%**, because by construction it has no efflux axis and must
assume P-gp-negative. Adding a second axis roughly halves it. It is the metric that translates a
statistical blind spot into a development decision: 93% of compounds that cannot reach the brain
would be progressed as though they can.

**Q: Which single result would you defend as novel?**
The 216-molecule label measurement and its decision-level consequence: 32.1% of B3DB BBB+
molecules are confirmed P-gp substrates (CI [24.5, 40.6]), the BBB label is not a usable proxy
for efflux status, and a single-axis model therefore leaks 93.3% of paradoxical compounds. The
literature says the mechanisms are distinct; nobody quantifies it at the *label* level or draws
out what it means for benchmark validity.

**Q: What would you do next, with six more months?**
Four things, in order. (1) Re-tune within the scaffold protocol so the random-vs-scaffold
comparison is clean. (2) Run 14 seeds to settle the BBB multi-task ordering. (3) Add BCRP/ABCG2
as a third barrier — the architecture already supports an arbitrary task list, and
`ARCHITECTURE.md` documents the extension path. (4) Validate the top repurposing leads in an in
vitro MDCK-MDR1 bidirectional permeability assay, which would turn computational candidates into
experimental ones.

**Q: Why keep `Initial Model/` if it has known defects?**
Provenance. It is the base-paper reference implementation, and modifying it would destroy the
baseline the rest of the work improves on. Its defects are documented rather than fixed —
`train_gnn.py` silently discards 269 of 1,890 molecules, `predict.py` cannot load its own
checkpoint, no metrics are persisted. The new `gnn/` package avoids all of them **by
construction**: one shared featuriser, checkpoints that describe their own architecture, and
every run writing metrics to disk.

**Q: How is this reproducible?**
Every stochastic step is seeded (`random_state=42`, ETKDGv3 seed 42, Vina `--seed 42`, all of
python/numpy/torch/CUDA in `set_all_seeds`). Every run writes a provenance JSON with arguments,
device, corpus statistics, split report and architecture. Checkpoints carry metadata read off
the live module. And four verification suites — 57 + 31 + 29 + 8 checks — assert that every
number quoted in prose matches the file it came from, because three manuscript defects existed
purely from copying figures between documents instead of re-reading the source.

---

## 12. Commands

```bash
# Single molecule, full explainability report
python "Novelty Upgrade/inference.py" --smiles "CN1CCC[C@H]1c2cccnc2" --explain

# Clinical benchmark suite (the 5 reference drugs)
python "Novelty Upgrade/inference.py" --test-benchmark

cd "Novelty Upgrade"

# 1. Featurise
python training/bbb_data_preprocessing.py
python training/pgp_data_preprocessing.py

# 2. Train
python training/bbb_model_training.py       # Optuna, 50 trials, scored on held-out validation
python training/pgp_model_training.py

# 3. Validate
python scripts/run_validation.py            # random + scaffold, 1000-resample bootstrap CIs
python scripts/run_ablation_study.py        # feature-subset ablation, 5-fold CV
python scripts/run_explainability_analysis.py

# 4. Apply
python scripts/run_repurposing_screen.py          # 2,405-drug pharmacopeia screen
python scripts/run_cryoem_docking_validation.py   # docking into PDB 6QEX

# 5. Multi-task GNN
python scripts/train_multitask_gnn.py --tasks bbb,pgp --split random
python scripts/train_multitask_gnn.py --tasks bbb        --split random   # single-task baseline
python scripts/train_multitask_gnn.py --tasks bbb,pgp --task-weights pgp=0 --tag sanity_pgpw0
python scripts/run_kfold_multitask.py --tasks bbb,pgp --folds 5 --tag multitask
python scripts/run_metric_analysis.py             # M1-M5 decision metrics
python scripts/run_seed_variance.py               # seed variance, MDE, P(A>B)

# 6. Verify  (run verify_doc_consistency.py after anything that changes a saved metric)
python scripts/verify_doc_consistency.py    # 57 checks
python scripts/verify_decision_metrics.py   # 31 checks
python scripts/verify_multitask_gnn.py      # 29 checks
python -m pytest tests/test_explainability.py   #  8 checks  (pytest file: needs -m pytest)
```
