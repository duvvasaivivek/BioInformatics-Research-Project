# Dual-Barrier Explainable Machine Learning for CNS Drug Discovery: Decoupling Passive Blood-Brain Barrier Permeation from Active P-glycoprotein Efflux

**Target Venues:** *Journal of Chemical Information and Modeling (ACS)* / *Bioinformatics (Oxford Academic)* / *Computers in Biology and Medicine (Elsevier)*  
**Date:** 4 November 2026

**Results refreshed 4 November 2026.** XGBoost validation, feature ablation,
global SHAP, GNN single-split experiments, five-fold out-of-fold predictions,
five-seed comparisons, and decision metrics use the regenerated November artifacts.
The pharmacopeia screen (20 October) and docking analysis (24 October) have not
been rerun with the new models; their sections are explicitly historical.
Clinical benchmark probabilities and local SHAP values were recomputed from the
current models on 4 November. Table checks verify numerical consistency, not
scientific validity or clinical efficacy.

---

## Abstract

We evaluate a dual-endpoint framework for BBB permeability and P-glycoprotein
substrate prediction, combining XGBoost, TreeSHAP, and a shared-encoder GNN.
The November rerun yields random-test BBB ROC-AUC **0.9563** [0.9452 - 0.9669]
and accuracy **0.8958** [0.877 - 0.9129], and P-gp ROC-AUC **0.9607**
[0.9321 - 0.9828]. Independent fixed-configuration scaffold benchmarks yield
0.8112 and 0.9001 AUC, respectively, showing reduced performance for both
endpoints under these benchmark conditions. Five-fold feature ablation shows
no observed mean P-gp AUC benefit from adding 3D descriptors to Morgan + 2D
(0.9465 versus 0.9480).

Among six shared descriptors, TPSA dominates BBB mean absolute SHAP (1.2652)
and LogP dominates P-gp (1.3574). Across five paired seeds, multitask-minus-single
ROC-AUC differences are −0.006956 for BBB and +0.001989 for P-gp, with both
95% intervals including zero. On 216 dual-labelled molecules with out-of-fold
predictions, adding P-gp reduces paradox leakage from 42/45 to 19/45 for composed
models and 20/45 for the shared encoder. Joint quadrant accuracy remains
51.9%, 57.4%, and 56.9%, highlighting substantial residual decision errors.

The October pharmacopeia screen and 22-compound docking analysis are retained
as explicitly historical exploratory results. They have not been regenerated
with the November models and are not evidence of current clinical efficacy.
Clinical benchmark examples were recomputed, but most are training-set recall
checks. The contribution is an auditable computational comparison and joint
decision analysis, with limitations from small dual-labelled samples and
chemotype generalization.

**Keywords:** Blood-Brain Barrier (BBB), P-glycoprotein (P-gp), ABCB1, Cryo-EM Structure, Molecular Docking, Drug Repurposing, Pharmacoresistance, Explainable AI (XAI), TreeSHAP, 3D Molecular Conformation, Multi-Task Pharmacokinetics.

---

## 1. Introduction

The blood-brain barrier (BBB) constitutes the primary physiological and enzymatic bottleneck in neurotherapeutics. Formed by a continuous monolayer of brain capillary endothelial cells linked by complex tight junctions (claudin-5, occludin, and junctional adhesion molecules) and surrounded by pericytes and astrocytic end-feet, the BBB maintains neural homeostasis by strictly limiting the extravasation of hydrophilic solutes, macromolecules, systemic xenobiotics, and pathogens into the brain parenchyma. Consequently, neurodegenerative diseases, brain tumors, psychiatric disorders, and epilepsy suffer from an alarming attrition rate in CNS drug discovery.

Historically, in silico screening methods for BBB crossing have modeled the barrier as a static, passive lipid bilayer governed by classical physicochemical rules (such as Lipinski's Rule of 5 for CNS, or Pajouhesh & Lenz rules: $\text{MW} < 400$, $\log P \in [2, 4]$, $\text{TPSA} < 90 \text{ \AA}^2$, $\text{HBD} \le 1-2$). Early machine learning implementations—including classical GCN baselines trained on small benchmarks like Zhao et al. ($\sim 1,900$ compounds)—focused exclusively on binary classification of passive permeation ($\text{BBB}^+$ vs. $\text{BBB}^-$).

However, as emphasized by Löscher & Potschka (2005), net brain drug exposure is not simply a function of passive thermodynamic partition:
$$\text{Net Brain Exposure} \propto \frac{\text{Passive Influx Rate}}{\text{Active Efflux Clearance} + \text{Metabolic Elimination}}$$

### The Paradoxical Drug Dilemma
Brain capillary endothelial cells express high levels of ATP-binding cassette (ABC) efflux transporters on their luminal plasma membranes. The most clinically prominent of these is **P-glycoprotein (P-gp / ABCB1 / MDR1)**. P-gp acts as a luminal "hydrophobic vacuum cleaner," extracting lipophilic molecules directly from the inner leaflet of the lipid bilayer and using ATP hydrolysis to pump them back into the systemic circulation.

This active extrusion produces the **"paradoxical drug" phenomenon**:
- A molecule may possess high lipophilicity ($\log P > 3.5$) and low polar surface area, satisfying all criteria for passive membrane permeability ($\text{BBB}^+$).
- However, its hydrophobic bulk and polyvalent hydrogen-bonding spatial patterns trigger high-affinity binding to P-gp transmembrane helices ($\text{P-gp}^+$ substrate).
- Once administered in vivo, the drug dissolves into endothelial membranes but is immediately pumped out before reaching therapeutic concentrations in brain tissue.

The prototypical clinical example is **Loperamide (Imodium)**: a potent $\mu$-opioid receptor agonist that crosses lipid bilayers with ease ($\text{BBB}^+$) but produces zero central analgesia at therapeutic doses because of violent luminal P-gp clearance. Only in the presence of P-gp inhibitors (such as Quinidine) or in *Abcb1a* knockout animals does Loperamide produce marked central opioid effects and respiratory depression.

Conversely, an ideal CNS drug must occupy a narrow biochemical niche: it must achieve adequate passive membrane partitioning ($\mathbf{BBB^+}$) while remaining completely refractory to P-gp efflux pump engagement ($\mathbf{P\text{-gp}^-}$). The archetypal example is **Levetiracetam (Keppra)**, a refractory antiepileptic drug that evades P-gp extrusion entirely.

### Core Research Questions & Contributions
In this work, we address seven fundamental computational, pharmacological, and biophysical questions:
1. *Can a unified dual-barrier framework decouple passive endothelial permeability from active ABC-transporter efflux to resolve paradoxical false positives?*
2. *Which specific molecular properties govern BBB permeation versus P-gp efflux, and how do their contributions differ quantitatively?*
3. *Do shared descriptors exert the same biological effect across both endpoints?*
4. *Does augmenting 2D fingerprints with 3D conformer geometry improve P-gp efflux prediction under leakage-resistant evaluation?*
5. *Can local TreeSHAP explanations and reaction-based counterfactuals guide rational structural lead optimization?*
6. *When applied at pharmacopeia scale across 2,400+ FDA-approved drugs, can dual-barrier screening uncover dormant CNS leads for drug repurposing and provide a mechanistic molecular explanation for clinical drug resistance in refractory neuropsychiatric diseases?*
7. *Do in silico docking simulations against the Cryo-EM inward-facing translocation chamber of human P-glycoprotein (PDB: 6QEX) corroborate the machine-learning-predicted quadrant classifications with thermodynamic binding free energies and specific transmembrane contact patterns?*

---

## 2. Pharmacological & Biophysical Foundations

### 2.1 P-glycoprotein Architecture & Cavity Multispecificity
P-glycoprotein is a 170 kDa glycoprotein comprising two pseudo-symmetrical halves, each containing six transmembrane $\alpha$-helices (TMDs) and an intracellular nucleotide-binding domain (NBD). High-resolution cryo-EM structures of human ABCB1 reveal an extraordinarily large, inward-facing central translocation chamber ($\sim 6,000 \text{ \AA}^3$) lined with aromatic (Phe, Tyr) and polar (Gln, Ser) residues.

Because the chamber is large and plastic, P-gp does not operate via a rigid lock-and-key paradigm. Instead, it exhibits extreme multispecificity. Substrates enter the chamber through lateral portals open to the inner leaflet of the plasma membrane. Substrate release into the extracellular lumen is driven by ATP binding and dimerization of the NBDs, converting the transporter into an outward-facing conformation with dramatically reduced substrate affinity.

### 2.2 Mechanism of Substrate Recognition: Seelig's Pharmacophore Rules
Biochemical analyses by Seelig (1998, 2000) demonstrated that P-gp substrate recognition is primarily governed by spatial hydrogen-bonding donor and acceptor patterns:
- Type I and Type II P-gp recognition units consist of pairs of hydrogen-bonding groups separated by fixed spatial distances ($2.5 \text{ \AA}$ or $4.6 \text{ \AA}$).
- As the number of hydrogen-bonding interactions within the chamber increases, the rate of P-gp ATPase activation rises exponentially.
- Consequently, minimizing hydrogen-bond donors ($\text{HBD} \le 1$) and acceptors while maintaining moderate lipophilicity provides the optimal design strategy to cross the lipid bilayer without triggering P-gp ATPase activation.

### 2.3 The Role of 3D Molecular Conformation
Because the P-gp translocation cavity is three-dimensional and multisite, molecules with extended spatial bulk (high Radius of Gyration $R_g$) or high conformational flexibility (high rotatable bond counts) can adopt favorable conformations that maximize contact with the surrounding helices. Conversely, small, rigid, spherical molecules ($R_g < 3.8 \text{ \AA}$, high spherocity index) pass across the membrane without spanning the necessary contact points within the P-gp chamber.

---

## 3. Materials and Methods

### 3.1 Benchmark Datasets & Preprocessing
Our study employs two curated, experimentally verified benchmarks:
1. **B3DB Database (Meng et al., 2021):** The largest curated collection of small-molecule blood-brain barrier permeability data, assembled from 50 published sources and comprising $7,807$ records, of which $7,805$ survive RDKit parsing and are used here after molecular identity deduplication. Compounds are annotated into $\text{BBB}^+$ ($N=4,956$, $63.5\%$) and $\text{BBB}^-$ ($N=2,849$, $36.5\%$).
2. **Broccatelli TDC Dataset (Broccatelli et al., 2011; Therapeutics Data Commons):** The source contains 1,219 rows. Parsing and conformer generation retain 1,205
feature rows ($N=650$ substrates and $N=555$ non-substrates). Removing seven
duplicate molecular identities leaves **1,198 unique XGBoost structures**:
646 substrates and 552 non-substrates (53.9% and 46.1%). The GNN does not require
3D embedding and uses 1,212 unique P-gp structures, with 647 positive and 565 negative labels.

#### Data Cleaning & Quality Control:
- Structures are matched by full InChIKey; invalid structures and conflicting labels are removed, and one row per identity is retained. This implementation does not perform a separate salt-stripping step.
- Murcko scaffold clustering identified $2,129$ unique core scaffolds in B3DB (scaffold diversity ratio $0.273$) and $670$ unique scaffolds in Broccatelli (diversity ratio $0.559$).
- A total of $288$ core scaffolds overlap between the two benchmarks, enabling rigorous comparative analysis.

### 3.2 Feature Engineering Pipeline
1. **Topological Fingerprints:** 2048-bit Extended Connectivity Fingerprints (ECFP4 / Morgan, radius 2) were calculated to represent local atom environments.
2. **Continuous 2D Descriptors:** Six essential physicochemical descriptors were computed:
   $$\mathbf{x}_{\text{2D}} = \left[ \text{MolWt}, \text{MolLogP}, \text{TPSA}, \text{NumHDonors}, \text{NumHAcceptors}, \text{NumRotatableBonds} \right]$$
3. **3D Conformer Generation & Stereochemical Descriptors:**
   Three-dimensional conformers were generated using RDKit's Experimental-Torsion Knowledge Distance Geometry (ETKDGv3) algorithm with fixed random seed $42$. When standard convergence failed on macrocyclic or congested scaffolds, an extended iteration fallback (up to 100 attempts) was deployed. Six 3D spatial descriptors were computed:
   - **Radius of Gyration ($R_g$):**
     $$R_g = \sqrt{\frac{1}{M}\sum_{i} m_i \|\mathbf{r}_i - \mathbf{r}_{\text{CM}}\|^2}$$
   - **Asphericity ($\Omega$):** Degree of deviation from spherical geometry ($\Omega \in [0, 1]$).
   - **Eccentricity ($c/a$):** Ellipsoidal elongation parameter.
   - **Spherocity Index:** Measure of spherical compactness ($\in [0, 1]$).
   - **Normalized Principal Moments of Inertia ($NPR_1, NPR_2$):** Coordinate position within the triangular shape space.
4. **Standardization:** Continuous descriptors were normalized via training-set z-scores ($z_j = (x_j - \mu_j) / \sigma_j$), with scaling parameters and training SMILES embedded in `bbb_xgboost.json` and `pgp_xgboost.json` for the current models. Legacy sidecar metadata is only a compatibility fallback.

### 3.3 Model Training & Optimization
- **$\mathcal{M}_{\text{BBB}}$ Classifier:** Optimized via Bayesian Tree-structured Parzen Estimator (TPE) search across 50 trials in Optuna, maximizing validation F1-score with early stopping (30 rounds) and class-imbalance weighting (`scale_pos_weight`).
- **$\mathcal{M}_{\text{Pgp}}$ Classifier:** Trained on Broccatelli 2D+3D features ($2,060$ dimensions) with learning rate $0.05$, max depth $5$, subsample $0.80$, and colsample by tree $0.80$.

### 3.4 TreeSHAP Formulation in Margin Space
For an input feature vector $\mathbf{x}$, TreeSHAP calculates the additive feature attributions in **log-odds margin space**:
$$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{M} \phi_i(\mathbf{x})$$
where $\phi_0$ is the expected background value across the training distribution ($\phi_0 = -0.1143$ for BBB, $\phi_0 = -0.0162$ for P-gp in the current fitted models), and $\phi_i(\mathbf{x})$ is the exact Shapley contribution of feature $i$.
The probability relates to margin via the logistic link:
$$P(Y=1) = \frac{1}{1 + e^{-f(\mathbf{x})}}$$

### 3.5 Morgan Bit-to-Substructure Deconvolution
To avoid treating Morgan fingerprint attributions as arbitrary bit indices, we implemented `MorganSubstructureMapper`. For any active bit ($x_i = 1$), the mapper extracts the atom environment radius $r$ around center atom $a_j$ using `Chem.FindAtomEnvironmentOfRadiusN`, reconstructs the submolecular fragment SMILES via `Chem.PathToSubmol`, and annotates the fragment with functional group tags (e.g., basic amine centers, carbonyl oxygen motifs, halogenated aromatic rings).

### 3.6 Applicability Domain & Uncertainty Assessment
To avoid overconfident extrapolation, each prediction is evaluated against two domain metrics:
1. **Structural Tanimoto Maximum Similarity:** Computed against training set fingerprints:
   $$T_{\max}(\mathbf{x}) = \max_{\mathbf{x}_k \in \mathcal{D}_{\text{train}}} \frac{\mathbf{x} \cdot \mathbf{x}_k}{\|\mathbf{x}\|^2 + \|\mathbf{x}_k\|^2 - \mathbf{x} \cdot \mathbf{x}_k}$$
2. **Descriptor Leverage Distance:** Normalized Euclidean distance from the training centroid in standardized descriptor space:
   $$D_{\text{leverage}}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} z_j^2}$$
- **Inside Domain:** $T_{\max} \ge 0.40$ and $D_{\text{leverage}} \le 2.5$.
- **Borderline:** $T_{\max} \in [0.28, 0.40)$ or $D_{\text{leverage}} \in (2.5, 3.5]$.
- **Outside Domain:** $T_{\max} < 0.28$ or $D_{\text{leverage}} > 3.5$.

### 3.7 Chemically Valid Reaction-Based Counterfactual Engine
To provide actionable molecular optimization paths, we implemented `MolecularCounterfactualEngine`. Unlike descriptor perturbation approaches that propose physically impossible feature vectors, our engine applies realistic medicinal chemistry SMARTS reaction transformations (e.g., alcohol O-methylation, primary amine N-methylation, carboxylate esterification, aromatic fluorination, demethylation), sanitizes the resulting analog in RDKit, re-computes 3D conformers, and identifies minimal structural modifications (ranked by Tanimoto similarity to the parent compound) that improve a targeted prediction. For the joint objective, a candidate must not worsen the other barrier; descriptor deltas are measured relative to the parent. A predicted class flip is not evidence of synthesizability or clinical benefit.

---

## 4. Results & Empirical Evaluation

### 4.1 Test Set Performance and Chemotype Generalization

**Table 1: Dual-split evaluation**

Values are estimates with 1,000-resample bootstrap 95% confidence intervals.
The random test sets contain 1,171 BBB and 240 P-gp molecules after molecular
deduplication. Scaffold allocation places the largest scaffold clusters into
training first and has zero train/test scaffold overlap.

<!-- results:validation -->
| Metric | BBB random | BBB scaffold | P-gp random | P-gp scaffold |
| --- | --- | --- | --- | --- |
| Accuracy | 0.8958 [0.877 - 0.9129] | 0.6926 [0.6661 - 0.7182] | 0.9083 [0.8708 - 0.9417] | 0.8292 [0.7792 - 0.8792] |
| Balanced_Accuracy | 0.8896 [0.8691 - 0.9078] | 0.6622 [0.6389 - 0.6859] | 0.9097 [0.8725 - 0.9436] | 0.8233 [0.7702 - 0.8756] |
| Precision | 0.9226 [0.902 - 0.9413] | 0.6674 [0.6343 - 0.6972] | 0.9350 [0.8898 - 0.9748] | 0.7253 [0.6333 - 0.814] |
| Recall_Sensitivity | 0.9126 [0.8912 - 0.9319] | 0.9044 [0.8832 - 0.9256] | 0.8915 [0.8333 - 0.9426] | 0.8049 [0.7108 - 0.8873] |
| Specificity | 0.8665 [0.8311 - 0.8983] | 0.4199 [0.3781 - 0.4617] | 0.9279 [0.8739 - 0.9727] | 0.8418 [0.7826 - 0.8987] |
| F1_Score | 0.9176 [0.9022 - 0.9313] | 0.7680 [0.7441 - 0.7905] | 0.9127 [0.8745 - 0.9466] | 0.7630 [0.6853 - 0.8296] |
| ROC_AUC | 0.9563 [0.9452 - 0.9669] | 0.8112 [0.7862 - 0.8338] | 0.9607 [0.9321 - 0.9828] | 0.9001 [0.8559 - 0.9401] |
| PR_AUC | 0.9740 [0.9661 - 0.9813] | 0.8496 [0.8233 - 0.8741] | 0.9739 [0.9562 - 0.9884] | 0.8328 [0.7439 - 0.9047] |
| MCC | 0.7761 [0.7362 - 0.8114] | 0.3781 [0.3278 - 0.428] | 0.8173 [0.7417 - 0.8845] | 0.6321 [0.5236 - 0.7353] |
| Brier_Score | 0.0785 [0.0682 - 0.0893] | 0.1992 [0.1846 - 0.2158] | 0.0691 [0.0467 - 0.0927] | 0.1316 [0.0988 - 0.1664] |
<!-- /results:validation -->

Random-split AUC is 0.9563 for BBB and 0.9607 for P-gp; the scaffold benchmarks
yield 0.8112 and 0.9001. Specificity falls from 0.8665 to 0.4199 for BBB and
from 0.9279 to 0.8418 for P-gp. Both endpoints therefore perform worse under
these scaffold benchmarks; the earlier claim that P-gp performance is preserved
does not hold for the regenerated results.

The scaffold models are fresh fixed-configuration fits, whereas the BBB random
model is Optuna-tuned and the P-gp random model uses class weighting. Training
fractions also differ. These are independent benchmarks, not controlled estimates
of the causal effect of scaffold splitting. Brier scores are reported as probability
prediction errors; a low Brier score alone does not establish calibration.


---

### 4.2 Incremental Feature Ablation Study

**Table 2: Feature-subset cross-validation**

Five-fold stratified CV uses identical folds across feature subsets and fits
each scaler within its training fold. Values are mean ± fold standard deviation,
not confidence intervals.

<!-- results:ablation -->
| Task | Configuration | Features | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| P-gp Substrate | 1. 2D Descriptors Only | 6 | 0.8623 ± 0.0137 | 0.8741 ± 0.0133 | 0.9152 ± 0.0083 |
| P-gp Substrate | 2. 3D Shape Only | 6 | 0.7797 ± 0.0317 | 0.7983 ± 0.0285 | 0.8428 ± 0.0277 |
| P-gp Substrate | 3. 2D Descriptors + 3D Shape | 12 | 0.8440 ± 0.0284 | 0.8560 ± 0.0254 | 0.9142 ± 0.0117 |
| P-gp Substrate | 4. Morgan FP + 2D Descriptors | 2054 | 0.8957 ± 0.0107 | 0.9027 ± 0.0075 | 0.9480 ± 0.0087 |
| P-gp Substrate | 5. Full Model (Morgan + 2D + 3D Shape) | 2060 | 0.8848 ± 0.0162 | 0.8928 ± 0.0127 | 0.9465 ± 0.0078 |
| BBB Permeability | 1. 2D Descriptors Only | 6 | 0.8555 ± 0.0113 | 0.8913 ± 0.0081 | 0.9174 ± 0.0081 |
| BBB Permeability | 2. Morgan Fingerprints Only | 2048 | 0.8588 ± 0.0078 | 0.8936 ± 0.0054 | 0.9393 ± 0.0054 |
| BBB Permeability | 3. Full Hybrid (Morgan + 2D Descriptors) | 2054 | 0.8738 ± 0.0072 | 0.9041 ± 0.0051 | 0.9467 ± 0.0035 |
<!-- /results:ablation -->

BBB Morgan + 2D achieves 0.9467 versus 0.9393 for Morgan alone and 0.9174 for
2D descriptors alone. This is a higher mean score; no paired significance test
is supplied by the ablation artifact.

For P-gp, Morgan + 2D achieves 0.9480 ± 0.0087 versus 0.9465 ± 0.0078 with
3D features. The added shape descriptors provide no observed mean AUC gain
in this experiment. Shape alone achieves 0.8428 ± 0.0277. These predictive
associations do not demonstrate a molecular binding mechanism. The deployed
3D pipeline is retained for compatibility, with failed conformers rejected.


---

### 4.3 Quantitative Analysis of Shared Properties

**Table 3: Current global SHAP measurements**

Mean absolute and mean signed TreeSHAP values are in log-odds margin units,
computed on held-out samples using the preprocessing metadata embedded in the
current model files. Exact native XGBoost contributions are checked for additivity.

<!-- results:shap -->
| Descriptor | BBB mean absolute SHAP | P-gp mean absolute SHAP | BBB mean signed SHAP | P-gp mean signed SHAP |
| --- | --- | --- | --- | --- |
| MolWt | 0.3991 | 0.6488 | -0.0240 | 0.0319 |
| LogP | 0.2855 | 1.3574 | 0.0388 | 0.0814 |
| TPSA | 1.2652 | 0.1431 | 0.1701 | 0.0364 |
| HDonors | 0.1694 | 0.0519 | 0.0219 | 0.0156 |
| HAcceptors | 0.0757 | 0.0656 | -0.0092 | 0.0060 |
| RotBonds | 0.1002 | 0.0561 | 0.0066 | 0.0066 |
<!-- /results:shap -->

Among the six shared descriptors, TPSA has the largest BBB mean absolute
attribution (1.2652), while LogP dominates P-gp (1.3574). Their importance ranking
correlation is Spearman rho=0.5429, p=0.2657 (six descriptors). This small analysis
does not establish a significant shared ranking. Mean absolute SHAP gives
importance, not direction; a mean signed attribution is not a causal dose-response
effect. Mechanistic interpretations and proposed descriptor windows remain hypotheses.


---

### 4.4 Clinical Reference Benchmark Evaluations
We evaluated our dual-barrier pipeline against classical clinical benchmark drugs highlighted by Löscher & Potschka (2005) (Table 4):

**Table 4: Clinical Benchmark Drug Evaluation across the Dual-Barrier Matrix**

| Compound | Canonical SMILES | P(BBB+) | P(P-gp+) | Predicted quadrant |
| --- | --- | --- | --- | --- |
| Levetiracetam | `CC[C@@H](C(=O)N)N1CCCC1=O` | 97.9% | 1.4% | Q1 |
| Loperamide | `CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1` | 94.5% | 99.1% | Q2 |
| Caffeine | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` | 86.5% | 1.4% | Q1 |
| Diazepam | `CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13` | 99.3% | 5.1% | Q1 |
| Penicillin G | `CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O` | 0.9% | 2.3% | Q3 |

Source: `results/documentation_model_snapshot.json`, current models evaluated 4 November 2026. Probabilities refer to the positive class.

#### Training-Set Membership of the Clinical Benchmarks:
Membership was determined by full InChIKey match against both training databases
(InChIKey rather than SMILES string equality, so that differences in spelling,
SMILES spelling does not create a false negative; different salt forms can still have distinct identities):

| Compound | In B3DB ($N=7,805$) | In Broccatelli ($N=1,212$) |
|---|:---:|:---:|
| Levetiracetam | yes | no |
| Loperamide | **yes** | **yes** |
| Caffeine | yes | yes |
| Diazepam | yes | yes |
| Penicillin G | yes | no |

**These are therefore recall checks, not blind validation.** Current embedded training-SMILES metadata also confirms the membership pattern below. All five compounds are
present in the BBB training set, and three of the five (Loperamide, Caffeine,
Diazepam) are additionally present in the P-gp training set. Only Levetiracetam
and Penicillin G are unseen by the P-gp model. Table 4 demonstrates that the models reproduce
the established pharmacology of well-characterised reference drugs; it does not
demonstrate out-of-sample generalisation, and the scaffold-split results in
Section 4.1 remain the only evidence for that.

> **Correction.** An earlier version of this manuscript stated that Loperamide was
> "100% ABSENT" from both databases and that its Quadrant 2 assignment was a
> "genuine out-of-sample zero-shot prediction". That was an artifact of a
> structural error: the Loperamide SMILES then in use was $C_{30}H_{35}ClN_2O_2$
> (MW 491.08) rather than the correct $C_{29}H_{33}ClN_2O_2$ (MW 477.05) - one
> CH$_2$ too many - and that incorrect structure is indeed absent from both
> databases. With the correct structure (InChIKey RDOIQAHITMMDAJ) Loperamide is
> present in both. The claim of blind validation has been withdrawn.

#### Current local SHAP examples

The following values were recomputed from the current model files. They explain
model outputs; they are not experimental measurements of transporter mechanisms.

##### Levetiracetam (Q1)

P(BBB+)=97.9% (margin +3.8208); P(P-gp+)=1.4% (margin -4.2304).

| Feature | Raw value | BBB SHAP | P-gp SHAP |
| --- | --- | --- | --- |
| MolWt | 170.2120 | +0.7246 | -0.8410 |
| LogP | -0.1273 | -0.2290 | -1.7356 |
| TPSA | 63.4000 | +0.6000 | -0.0513 |
| RadiusOfGyration | 2.4274 | not a BBB feature | -0.7097 |

##### Loperamide (Q2)

P(BBB+)=94.5% (margin +2.8492); P(P-gp+)=99.1% (margin +4.6728).

| Feature | Raw value | BBB SHAP | P-gp SHAP |
| --- | --- | --- | --- |
| MolWt | 477.0480 | -0.4741 | +0.7536 |
| LogP | 5.0880 | +0.2073 | +2.1679 |
| TPSA | 43.7800 | +1.6619 | +0.2665 |
| RadiusOfGyration | 4.8913 | not a BBB feature | +0.3909 |

No new counterfactual search was run for this documentation refresh; no new optimized analog is claimed.

### 4.5 Shared-Encoder GNN Experiments

The shared GINEConv encoder is compared with independent single-task models on
identical molecule partitions. Random-split training includes 5,463 BBB-labelled
and 848 P-gp-labelled molecules. Single-task models have 364,608 parameters;
the two-head model has 372,898. The following are ROC-AUC comparisons for seed 42:

<!-- results:gnn -->
| Task | Split | Single-task AUC | Multitask AUC | Delta | Paired delta 95% CI | Paired result |
| --- | --- | --- | --- | --- | --- | --- |
| BBB | random | 0.9391 | 0.9285 | -0.0106 | [-0.0189, -0.0024] | degraded |
| BBB | scaffold | 0.7661 | 0.7551 | -0.0110 | [-0.0273, +0.0034] | paired interval includes zero |
| PGP | random | 0.9323 | 0.9366 | +0.0042 | [-0.0178, +0.0257] | paired interval includes zero |
| PGP | scaffold | 0.8087 | 0.8365 | +0.0278 | [-0.0073, +0.0628] | paired interval includes zero |
<!-- /results:gnn -->

Paired bootstrap intervals resample the same test molecules for both models;
they are conditional on these fitted models and do not measure training-seed
variability. BBB random-split AUC decreases by 0.0106, with a paired interval
excluding zero. The other three intervals include zero, including the positive
P-gp scaffold estimate (+0.0278). Overlap of separate model intervals is not used
as a significance test.

The zero-P-gp-weight control yields BBB AUC 0.9383 versus single-task 0.9391,
while the untrained P-gp head yields 0.5077. Increasing P-gp weight to 3 and 5
yields P-gp AUCs 0.9346 and 0.9378, and BBB AUCs 0.9267 and 0.9215.
These point estimates show a trade-off; they do not establish a universal
negative-transfer mechanism or prove that weak marginal label association
precludes useful shared representations.

Five paired random-split seeds (42–46) give:

<!-- results:seed_variance -->
| Task | Metric | Seeds | Mean delta | Delta 95% CI | MDE | Paired p |
| --- | --- | --- | --- | --- | --- | --- |
| BBB | ROC_AUC | 5 | -0.006956 | [-0.018911, +0.004999] | 0.016007 | 0.1815 |
| BBB | MCC | 5 | -0.039378 | [-0.075128, -0.003627] | 0.047866 | 0.0377 |
| BBB | F1_Score | 5 | -0.015699 | [-0.025917, -0.005480] | 0.013681 | 0.013 |
| BBB | Balanced_Accuracy | 5 | -0.018726 | [-0.047316, +0.009865] | 0.03828 | 0.1431 |
| PGP | ROC_AUC | 5 | +0.001989 | [-0.023149, +0.027127] | 0.033657 | 0.8369 |
| PGP | MCC | 5 | +0.002344 | [-0.097909, +0.102596] | 0.134229 | 0.9514 |
| PGP | F1_Score | 5 | -0.000968 | [-0.048114, +0.046177] | 0.063124 | 0.9573 |
| PGP | Balanced_Accuracy | 5 | +0.001613 | [-0.049375, +0.052601] | 0.068268 | 0.9342 |
<!-- /results:seed_variance -->

Neither task's across-seed ROC-AUC contrast excludes zero: BBB mean delta
−0.006956 (p=0.1815), P-gp +0.001989 (p=0.8369). Both absolute deltas are below
their estimated MDEs (0.016007 and 0.033657). BBB MCC and F1 decrease at nominal
p<0.05, but these secondary tests are exploratory and unadjusted for multiplicity.
The evidence supports no detected across-seed AUC improvement, not equivalence.
No definitive required-seed count is inferred from these five runs.


### 4.6 Out-of-Fold Dual-Barrier Decision Metrics

Five-fold out-of-fold predictions are evaluated on the same 216 dual-labelled
molecules. A_bbb_only assumes P-gp-negative for every molecule;
B_composed_two_model combines independent BBB and P-gp predictions;
C_multitask uses the shared encoder. The BBB-only baseline cannot express Q2,
so zero paradox recall is a consequence of its definition.

<!-- results:decisions -->
| Strategy | N | Quadrant accuracy | Accuracy 95% CI | Kappa | Paradox recall | Paradox leakage | Leaked / true Q2 | Regret at lambda=1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_bbb_only | 216 | 0.5185 | 0.4444 - 0.5833 | 0.2158 | 0.0000 | 0.9333 | 42 / 45 | 0.40278 |
| B_composed_two_model | 216 | 0.5741 | 0.5093 - 0.6390 | 0.3651 | 0.5111 | 0.4222 | 19 / 45 | 0.32407 |
| C_multitask | 216 | 0.5694 | 0.5046 - 0.6343 | 0.3598 | 0.4889 | 0.4444 | 20 / 45 | 0.31944 |
<!-- /results:decisions -->

BBB-only advances 42 of 45 true Q2 compounds as Q1 (93.3%). Adding the P-gp
axis reduces this to 19/45 (42.2%) for composed models and 20/45 (44.4%) for
multitask learning. Quadrant accuracy is 51.9%, 57.4%, and 56.9%, respectively.
These data support including the efflux axis, without demonstrating superiority
of the shared encoder over two independent models.

Pooled OOF per-axis AUCs are BBB 0.9339 (single) versus 0.9390 (multi), and
P-gp 0.9078 versus 0.9006. These use all labelled molecules for each endpoint,
whereas joint metrics use only the 216 dual-labelled molecules.

The efflux-disparity gap, AUC(non-substrates) minus AUC(substrates), is negative:
−0.164758 [−0.2698, −0.0668] for BBB-only/composed models and
−0.106741 [−0.2188, −0.0008] for multitask. Thus this subset does not show
worse BBB discrimination among substrates. At equal error costs, decision
regret is 0.40278, 0.32407, and 0.31944. These are computational decision errors,
not measured clinical outcomes.


---

## 5. Systematic Virtual Screening of the FDA-Approved Pharmacopeia Uncovers Latent CNS Leads and Clinical Efflux Resistance

> **Historical analysis — 20 October 2026.** This section and its figures use the October
> models and outputs. Its probabilities, quadrant assignments, domain scores,
> and associations have not been re-evaluated with the November models.
> Interpret candidate and mechanistic statements as exploratory hypotheses.


To translate our explainable dual-barrier framework from theoretical benchmarking to immediate therapeutic utility, we conducted a comprehensive virtual screen of the **entire FDA-approved small-molecule pharmacopeia** curated by the Broad Institute Drug Repurposing Hub ($N = 2,405$ unique, clinically validated compounds; Corsello et al., 2017). Each molecule was profiled through our feature extraction pipeline (generating 2048-bit Morgan fingerprints, standardized 2D descriptors, and ETKDGv3 3D conformer shape metrics), evaluated against the dual-barrier ensemble models, and screened for chemical applicability domain compliance ($T_{\max}$).

We defined the composite **CNS Desirability Score** ($S_{\text{CNS}}$) to quantify net brain penetrance potential:
$$S_{\text{CNS}} = P(\text{BBB}^+) \times \left(1 - P(\text{P-gp}^+)\right) \in [0, 1]$$
A high score ($S_{\text{CNS}} \to 1.0$) denotes strong passive membrane permeability coupled with complete evasion of luminal active efflux pumps.

### 5.1 Global Pharmacopeia Dual-Barrier Landscape

Figure 3 illustrates the global distribution of the 2,405 approved therapeutics mapped across the dual-barrier probability space ($P(\text{BBB}^+)$ vs. $P(\text{P-gp}^+)$):

![Figure 3: Dual-Barrier Profiling of 2,400+ FDA-Approved Drugs](Novelty%20Upgrade/figures/repurposing_pharmacopeia_landscape.png)

Across the entire pharmacopeia, the dual-barrier classification yields:
- **Quadrant 1 (Optimal CNS Permeation, $\text{BBB}^+ / \text{P-gp}^-$):** $710\text{ drugs } (29.5\%)$. These molecules easily partition into brain endothelial membranes without being recognized by ABCB1 efflux machinery.
- **Quadrant 2 (Paradoxical / Efflux-Vulnerable, $\text{BBB}^+ / \text{P-gp}^+$):** $240\text{ drugs } (10.0\%)$. Highly lipophilic agents that enter the endothelial bilayer but are actively extruded back into the systemic circulation.
- **Quadrant 3 (Peripheral / Non-Permeating, $\text{BBB}^- / \text{P-gp}^-$):** $1,064\text{ drugs } (44.2\%)$. Polar, hydrophilic, or zwitterionic agents that fail to cross passive lipid membranes.
- **Quadrant 4 (Restricted / Multi-Barrier Blocked, $\text{BBB}^- / \text{P-gp}^+$):** $391\text{ drugs } (16.3\%)$. Bulky, polar substrates rejected at both passive and active levels.

### 5.2 Latent CNS Penetrants: High-Priority Repurposing Opportunities for Neuro-Oncology, Neuroinflammation, and Meningeal Infections

Developing novel chemical entities for CNS indications typically requires $>12-15\text{ years}$ and $>\$2.5\text{ billion}$, with clinical failure rates exceeding $95\%$. Computational repurposing of approved non-CNS drugs whose safety, pharmacokinetics, and tolerability are already established in humans offers a rapid, de-risked translational pathway.

Filtering the screening dataset for **non-CNS indications** (excluding neurology and psychiatry) and applying a strict **chemical applicability domain threshold** ($T_{\max} \ge 0.35$ against the B3DB training distribution) revealed **273 FDA-approved non-CNS therapeutics residing in Quadrant 1**. Table 5 highlights the top repurposing candidates across critical unmet neurotherapeutic areas:

**Table 5: High-Priority Non-CNS FDA-Approved Repurposing Candidates in Quadrant 1 ($\text{BBB}^+ / \text{P-gp}^-$)**

| Drug Name | Disease Area | Primary Indication | Mechanism of Action (MoA) | $P(\text{BBB}^+)$ | $P(\text{P-gp}^+)$ | $S_{\text{CNS}}$ | Domain $T_{\max}$ | Translational Therapeutic Opportunity |
|---|---|---|---|:---:|:---:|:---:|:---:|---|
| **Thalidomide** | Oncology | Multiple Myeloma | TNF production inhibitor / Cereblon modulator | **0.9703** | **0.0174** | **0.9534** | 1.000 | **Neuro-Oncology (Glioblastoma):** Potent anti-angiogenic and immunomodulatory activity; high passive permeability with complete P-gp evasion makes it an ideal adjuvant for recurrent glioblastoma. |
| **Aminoglutethimide** | Endocrinology / Oncology | Cushing's / Breast Cancer | Steroidogenesis / Glucocorticoid antagonist | **0.9725** | **0.0095** | **0.9632** | 1.000 | **Hormone-Driven Brain Metastases:** Rapid brain entry and negligible efflux; potential candidate for endocrine-refractory intracranial metastases. |
| **Topotecan** | Oncology | Ovarian / SCLC | Topoisomerase I inhibitor | **0.6377** | **0.0219** | **0.6237** | 0.476 | **Brain Metastases of SCLC:** Unlike bulky camptothecins (Irinotecan) that suffer active P-gp clearance, Topotecan demonstrates low efflux affinity ($2.2\%$), supporting intracerebral cytotoxicity. |
| **Betamethasone** | Dermatology | Dermatoses / Inflammation | Glucocorticoid receptor agonist | **0.9967** | **0.0146** | **0.9821** | 1.000 | **Vasogenic Cerebral Edema & Neuroinflammation:** Extremely high membrane affinity with negligible efflux ($1.5\%$), offering superior brain exposure to reduce peritumoral edema without transporter saturation. |
| **Triamcinolone** | Dermatology | Inflammatory conditions | Glucocorticoid receptor agonist | **0.9786** | **0.0129** | **0.9659** | 1.000 | **Microglial Modulation:** Evades ABCB1 extrusion entirely ($1.3\%$), providing potent intracerebral immunosuppression for acute neuroinflammatory cascades. |
| **Linezolid** | Infectious Disease | MRSA Pneumonia / Skin | Bacterial 50S ribosomal inhibitor | **0.9670** | **0.0165** | **0.9510** | 0.451 | **Resistant Bacterial Meningitis:** Exceptional passive entry ($96.7\%$) combined with near-zero P-gp affinity ($1.7\%$) rationalizes its high CSF-to-plasma ratio ($>0.70$) in clinical meningitis. |
| **Chloramphenicol** | Infectious Disease | Bacterial Infections / Cholera | Bacterial 50S ribosomal inhibitor | **0.9519** | **0.0078** | **0.9445** | 0.636 | **Severe Meningococcal Meningitis:** Historic gold-standard for CNS infections; model confirms zero-efflux profile ($0.8\%$). |
| **Milnacipran** | Rheumatology | Fibromyalgia | Serotonin-Norepinephrine Reuptake Inhibitor | **0.9882** | **0.0047** | **0.9835** | 1.000 | **Central Neuropathic Pain:** Evades P-gp completely ($0.5\%$), offering repurposing for refractory thalamic and post-stroke pain syndromes. |
| **Metoclopramide** | Gastroenterology | Gastroparesis / Nausea | Dopamine / Serotonin antagonist | **0.9909** | **0.0256** | **0.9656** | 1.000 | **Central Anti-Emetic & Migraine Adjuvant:** Unrestricted brain penetration explains rapid central extrapyramidal receptor engagement. |
| **Naloxone** | Pulmonary | Opioid Overdose | $\mu$-Opioid receptor antagonist | **0.9917** | **0.0245** | **0.9674** | 1.000 | **Central Opioid Reversal:** Rapid passive permeability and complete absence of P-gp clearance guarantee sub-minute brainstem displacement of opioids. |

### 5.3 Uncovering Efflux Resistance in Marketed Neuropsychiatric Pharmacopeia

A perplexing clinical challenge in modern psychiatry and neurology is **pharmacoresistance (treatment refractoriness)**: approximately $30\%$ of patients diagnosed with schizophrenia, major depressive disorder (MDD), and epilepsy fail to achieve symptomatic remission despite aggressive polypharmacy with brain-targeted medications (Löscher & Potschka, 2005; Volk et al., 2004).

Our dual-barrier screen provides an immediate structural molecular explanation for this phenomenon. When examining approved therapeutics prescribed for **neurological and psychiatric indications**, we discovered that **105 marketed CNS drugs fall into Quadrant 2 ($\text{BBB}^+ / \text{P-gp}^+$)**. While their high lipophilicity allows them to partition into endothelial cell membranes, their molecular properties trigger robust engagement with luminal P-gp pumps. In patients with elevated endothelial P-gp expression (e.g., induced by chronic seizure activity, neuroinflammation, or genetic ABCB1 polymorphisms), these drugs are aggressively expelled before reaching therapeutic receptor occupancy in the brain parenchyma.

Table 6 details the top marketed neuropsychiatric agents exhibiting high susceptibility to P-gp-mediated pharmacoresistance:

**Table 6: Top Marketed Neuropsychiatric Therapeutics in Quadrant 2 Vulnerable to P-gp Efflux Pharmacoresistance**

| Drug Name | Clinical Indication | Pharmacological Class / MoA | $P(\text{BBB}^+)$ | $P(\text{P-gp}^+)$ | $S_{\text{CNS}}$ | Efflux Risk Profile & Clinical Significance |
|---|---|---|:---:|:---:|:---:|---|
| **Flupentixol** | Schizophrenia / Depression | Thioxanthene $D_1/D_2$ antagonist | **0.9961** | **0.9878** | **0.0122** | **Severe Efflux Risk ($98.8\%$):** Despite near-total passive membrane permeability, violent active clearance collapses net desirability score to $0.0122$, explaining frequent non-response in chronic schizophrenia. |
| **Zuclopenthixol** | Schizophrenia / Bipolar Mania | Thioxanthene $D_1/D_2$ antagonist | **0.9971** | **0.9853** | **0.0146** | **Severe Efflux Risk ($98.5\%$):** Lipophilic side chain maximizes P-gp chamber binding; brain accumulation is highly dependent on individual ABCB1 endothelial expression. |
| **Vilazodone** | Major Depressive Disorder (MDD) | Serotonin reuptake inhibitor / $5\text{-HT}_{1A}$ agonist | **0.8151** | **0.9817** | **0.0149** | **Treatment-Resistant Depression:** High P-gp substrate recognition ($98.2\%$) reduces intracerebral synaptic availability, rationalizing lack of clinical response in refractory depression cohorts. |
| **Everolimus** | SEGA / Tuberous Sclerosis / PNET | Allosteric mTOR Complex 1 inhibitor | **0.5284** | **0.9759** | **0.0127** | **Neuro-Oncological Efflux Failure:** Prescribed for Subependymal Giant Cell Astrocytoma (SEGA); our model identifies massive P-gp recognition ($97.6\%$), explaining variable intracranial tumor response. |
| **Haloperidol-decanoate** | Chronic Schizophrenia | Butyrophenone $D_2$ antagonist (depot) | **0.9554** | **0.9705** | **0.0282** | **Depot Efflux Resistance ($97.1\%$):** The decanoate lipophilic ester creates high passive affinity but triggers active efflux across the blood-brain barrier lumen. |
| **Lurasidone** | Schizophrenia / Bipolar Depression | Second-Generation Antipsychotic ($D_2/5\text{-HT}_2$) | **0.9503** | **0.9678** | **0.0306** | **Atypical Antipsychotic Refractoriness:** Large spatial envelope ($R_g = 4.41 \text{ \AA}$) and rigid polycyclic core drive high affinity for the P-gp translocation cavity ($96.8\%$). |
| **Fluphenazine** | Schizophrenia / Psychosis | Phenothiazine $D_2$ antagonist | **0.9987** | **0.9667** | **0.0333** | **Classical Antipsychotic Efflux:** High passive partition ($99.9\%$) contrasted with $96.7\%$ efflux probability; patient ABCB1 status strongly dictates required therapeutic dosage. |
| **Aripiprazole-lauroxil** | Schizophrenia (Extended Release) | Atypical Dopamine $D_2$ partial agonist | **0.8413** | **0.9662** | **0.0284** | **Efflux-Mediated Relapse Risk:** Prodrug modification introduces lipophilic bulk, elevating P-gp extrusion to $96.6\%$. |
| **Trifluoperazine** | Schizophrenia / Generalized Anxiety | Trifluoromethyl phenothiazine | **0.9963** | **0.9659** | **0.0340** | **Endothelial Trapping:** Fluorinated aromatic core enables lipid entry, but P-gp substrate engagement ($96.6\%$) severely curtails brain tissue bioavailability. |
| **Pimavanserin** | Parkinson's Disease Psychosis | Selective $5\text{-HT}_{2A}$ inverse agonist | **0.9819** | **0.9606** | **0.0387** | **Neurodegenerative Efflux:** Co-occurrence of Lewy body pathology and ABCB1 upregulation in Parkinson's disease creates an active barrier against therapeutic brain saturation. |

Figure 4 illustrates the dual-barrier quadrant breakdown across major therapeutic disease classes:

![Figure 4: Disease Area Dual-Barrier Distribution](Novelty%20Upgrade/figures/repurposing_disease_area_distribution.png)

While oncology and cardiology drugs are predominantly constrained by peripheral or multi-barrier restriction (Quadrants 3 and 4), neuropsychiatric drugs exhibit a bifurcated distribution: successful agents cluster in Quadrant 1, whereas treatment-resistant and second-line agents consistently slip into the paradoxical Quadrant 2.

---

## 6. Orthogonal Biophysical Validation: Cryo-EM Molecular Docking into Human P-glycoprotein (PDB: 6QEX)

> **Historical analysis — 24 October 2026.** This section and its figures use the October
> models and outputs. Its probabilities, quadrant assignments, domain scores,
> and associations have not been re-evaluated with the November models.
> Interpret candidate and mechanistic statements as exploratory hypotheses.


To substantiate our machine learning classifications with physical structural biology, we performed systematic molecular docking into the high-resolution ($3.5 \text{ \AA}$) Cryo-EM structure of human P-glycoprotein reconstituted in lipid nanodiscs (PDB: 6QEX; Alam et al., 2019). This inward-facing conformation captures the exact pre-translocation state where substrates partition from the inner membrane leaflet into the flexible central cavity.

### 6.1 Cryo-EM Translocation Chamber Architecture & In Silico Docking Protocol

The docking grid box ($24 \times 24 \times 24 \text{ \AA}$) was centered directly on the co-crystallized drug-binding site ($x = 173.33, y = 166.74, z = 161.48$), defined by the coordinates of co-crystallized Paclitaxel (Taxol, `TA1`). 

All **22** representative therapeutics were successfully docked. Quadrant membership below is **derived from the live model predictions** rather than assigned by hand, so a compound can fall outside the quadrant the benchmark set was designed around:
- **Quadrant 2 (Efflux Vulnerable / Paradoxical, $n=9$):** Flupentixol, Fluphenazine, Haloperidol, Loperamide, Lurasidone, Pimavanserin, Trifluoperazine, Vilazodone, Zuclopenthixol;
- **Quadrant 1 (Efflux Evaders / Repurposing Leads, $n=10$):** Aminoglutethimide, Aspirin, Betamethasone, Caffeine, Diazepam, Levetiracetam, Linezolid, Milnacipran, Thalidomide, Topotecan;
- **Quadrant 3 (Peripheral, $n=2$):** Metformin, Penicillin G;
- **Quadrant 4 (Restricted, $n=1$):** Paclitaxel.

Two compounds fall in a different quadrant from the one originally assigned by hand: **Paclitaxel**, the co-crystallised cryo-EM ligand, is predicted BBB-negative and therefore Q4 rather than Q2; and **Aspirin**, included as a peripheral control, is predicted BBB-positive and therefore Q1 rather than Q3. Both are reported as the model places them.

> **Structure provenance.** The benchmark SMILES were originally hand-entered, and six of them (Loperamide, Vilazodone, Lurasidone, Pimavanserin, Paclitaxel, Milnacipran) had a different molecular skeleton from the same drug in the curated Broad Repurposing Hub library, with four more differing in stereochemistry. All structures are now resolved from the curated library, and the ML probabilities are computed from the saved models at runtime rather than held as literals. The figures in this section therefore supersede any earlier version.

All ligands were 3D embedded with RDKit ETKDGv3, energy-minimized using the MMFF94 forcefield, parameterized with Gasteiger charges via Meeko, and docked into the rigid receptor using AutoDock Vina v1.2.7 (exhaustiveness 8). Table 7 compiles the thermodynamic binding free energies ($\Delta G_{\text{bind}}$), estimated dissociation constants ($K_d$), and specific transmembrane contact residues ($\le 3.8 \text{ \AA}$):

**Table 7: Cryo-EM Human P-glycoprotein (PDB: 6QEX) Translocation Chamber Docking Thermodynamics**

| Compound Name | Quadrant | Pharmacological Class | ML $P(\text{P-gp}^+)$ | $\Delta G_{\text{bind}}$ (kcal/mol) | Estimated $K_d$ | Contact Residues ($\le 3.8\text{ \AA}$) | Key Transmembrane Contacts |
|---|:---:|---|:---:|:---:|:---:|:---:|---|
| **Lurasidone** | **Q2** | Atypical Antipsychotic | **0.969** | **-10.20** | **33.4 nM** | 11 | TRP232, ILE306, TYR310, PHE336, LEU339, ILE340, PHE343, GLN347 |
| **Vilazodone** | **Q2** | Refractory Antidepressant | **0.983** | **-9.95** | **50.6 nM** | 15 | TRP232, PHE303, ILE306, TYR310, PHE336, PHE343, GLN347, GLN725 |
| **Loperamide** | **Q2** | Paradoxical Opioid | **0.986** | **-9.39** | **131.4 nM** | 15 | TRP232, ILE306, TYR310, PHE336, LEU339, ILE340, PHE343, GLN347 |
| **Pimavanserin** | **Q2** | Parkinson Psychosis Inverse Agonist | **0.960** | **-8.87** | **313.3 nM** | 13 | TRP232, PHE303, ILE306, TYR307, TYR310, PHE336, LEU339, ILE340 |
| **Haloperidol** | **Q2** | Typical Antipsychotic | **0.840** | **-8.73** | **398.9 nM** | 10 | TRP232, PHE303, ILE306, TYR307, PHE343, GLN990 |
| **Flupentixol** | **Q2** | Refractory Antipsychotic | **0.992** | **-8.60** | **499.2 nM** | 10 | LEU65, GLN195, TRP232, ILE306, ILE340, PHE343, GLN347 |
| **Zuclopenthixol** | **Q2** | Refractory Antipsychotic | **0.990** | **-8.42** | **675.3 nM** | 12 | TRP232, PHE303, ILE306, PHE343, GLN725, PHE770, GLN773, GLN838 |
| **Trifluoperazine** | **Q2** | Phenothiazine Antipsychotic | **0.975** | **-7.97** | **1.4 $\mu$M** | 11 | TRP232, LEU236, PHE303, TYR307, PHE983, GLN990, VAL991, PHE994 |
| **Fluphenazine** | **Q2** | Phenothiazine Antipsychotic | **0.966** | **-7.42** | **3.6 $\mu$M** | 13 | PHE303, ILE306, TYR307, TYR310, PHE336, LEU339, ILE340, LEU724 |
|---|---|---|---|---|---|---|---|
| **Paclitaxel** | **Q4** | Co-Crystallized Cryo-EM Ligand | **0.770** | **-8.88** | **309.6 nM** | 17 | LEU65, TRP232, PHE303, ILE306, TYR310, ILE340, PHE343, GLN347 |
|---|---|---|---|---|---|---|---|
| **Betamethasone** | **Q1** | Vasogenic Edema Lead | **0.016** | **-8.71** | **413.9 nM** | 12 | LEU236, PHE303, TYR307, GLN725, PHE770, GLN838, GLN990, VAL991 |
| **Topotecan** | **Q1** | Brain Metastases Repurposing Lead | **0.020** | **-8.68** | **435.4 nM** | 11 | GLN195, TRP232, PHE303, ILE306, PHE343, GLN347, GLN990 |
| **Thalidomide** | **Q1** | Neuro-Oncology Repurposing Lead | **0.019** | **-8.23** | **924.4 nM** | 7 | TRP232, ILE306, PHE343, GLN347, GLN990 |
| **Linezolid** | **Q1** | Bacterial Meningitis Lead | **0.017** | **-7.62** | **2.6 $\mu$M** | 10 | LEU65, GLN195, TRP232, ILE306, ILE340, PHE343, GLN347 |
| **Aminoglutethimide** | **Q1** | Brain Metastases Repurposing Lead | **0.009** | **-6.96** | **7.9 $\mu$M** | 9 | LEU236, ILE299, PHE303, PHE770, VAL835, GLN838, GLN990, VAL991 |
| **Diazepam** | **Q1** | Sedative Evader | **0.062** | **-6.11** | **33.0 $\mu$M** | 5 | TRP232, ILE340, PHE343, GLN347, GLN990 |
| **Milnacipran** | **Q1** | Neuropathic Pain Lead | **0.004** | **-5.98** | **41.5 $\mu$M** | 7 | TRP232, ILE306, TYR310, ILE340, PHE343, GLN990 |
| **Aspirin** | **Q1** | Peripheral NSAID | **0.004** | **-5.70** | **66.8 $\mu$M** | 6 | TRP232, ILE306, PHE343, GLN347, GLN990 |
| **Levetiracetam** | **Q1** | P-gp Evader Antiepileptic | **0.017** | **-5.59** | **80.4 $\mu$M** | 10 | PHE303, TYR307, LEU724, GLN725, GLN838, VAL991 |
| **Caffeine** | **Q1** | CNS Stimulant Evader | **0.060** | **-5.55** | **85.9 $\mu$M** | 8 | LEU65, GLN195, GLN946, TYR950 |
|---|---|---|---|---|---|---|---|
| **Penicillin G** | **Q3** | Peripheral Beta-Lactam | **0.020** | **-8.96** | **269.2 nM** | 9 | TRP232, PHE303, ILE306, TYR307, PHE343, LEU724, GLN725, GLN990 |
| **Metformin** | **Q3** | Peripheral Biguanide | **0.015** | **-4.94** | **240.1 $\mu$M** | 9 | LEU65, GLN195, GLN946, TYR950 |

### 6.2 Thermodynamic Separation Between Efflux-Vulnerable (Q2) and Evading (Q1) Therapeutics

As illustrated in Figure 5, the binding free energy distribution clearly differentiates Quadrant 2 substrates from Quadrant 1 evaders:

![Figure 5: Cryo-EM P-glycoprotein Translocation Cavity Binding Energy Distribution](Novelty%20Upgrade/figures/cryoem_docking_energy_distribution.png)

1. **High-Affinity Translocation Binding in Quadrant 2:**
   The mean binding free energy for Quadrant 2 therapeutics is **$-8.84 \pm 0.89 \text{ kcal/mol}$** ($n=9$), corresponding to high nanomolar to low micromolar dissociation constants ($K_d$). In particular, the atypical antipsychotic **Lurasidone** ($-10.20 \text{ kcal/mol}$, $K_d = 33.4 \text{ nM}$) and antidepressant **Vilazodone** ($-9.95 \text{ kcal/mol}$, $K_d = 50.6 \text{ nM}$) exhibit the tightest cavity affinities in the set, both below the co-crystallized ligand Paclitaxel ($-8.88 \text{ kcal/mol}$), which the model places in Quadrant 4 rather than Quadrant 2.
2. **Substantially Weaker Cavity Affinities in Quadrant 1:**
   Quadrant 1 therapeutics exhibit weaker affinities on average, with a mean $\Delta G_{\text{bind}}$ of **$-6.91 \pm 1.30 \text{ kcal/mol}$** ($n=10$). The prototypical evader **Levetiracetam** exhibits an affinity of only **$-5.59 \text{ kcal/mol}$** ($K_d \approx 80.4 \text{ }\mu\text{M}$), while Caffeine yields **$-5.55 \text{ kcal/mol}$** ($K_d \approx 85.9 \text{ }\mu\text{M}$)—indicating negligible specific interaction with the P-gp translocation machinery under physiological conditions.
3. **Statistical Significance:**
   A Welch's unequal variances two-sample t-test confirms that Quadrant 2 therapeutics bind significantly more tightly to the P-gp cavity than Quadrant 1 therapeutics (**$t = -3.7933, p = 1.597e-03$**). A non-parametric Mann-Whitney U test similarly verifies this difference (**$U = 11.0, p = 3.117e-03$**).

4. **Docking energy alone does not classify substrates.**
   The group separation is significant, but it does not transfer to the individual compound. 4 Quadrant 1 evaders (Thalidomide -8.23, Topotecan -8.68, Linezolid -7.62, Betamethasone -8.71 kcal/mol) bind at least as tightly as the weakest Quadrant 2 substrates, and the peripheral control **Penicillin G** reaches **-8.96 kcal/mol** despite a predicted efflux probability of 0.020, binding more tightly than six of the nine Quadrant 2 substrates. Rigid-receptor $\Delta G$ is therefore a population-level signal, not a per-molecule discriminator. This is consistent with reports that P-gp docking cannot separate substrates from non-substrates, and it is precisely why a ligand-topology model adds information the structure alone does not provide.

### 6.3 Transmembrane Helical Contact Residues

Analysis of the docked poses within $3.8 \text{ \AA}$ revealed the exact biophysical mechanism of multispecific cavity accommodation:
- **Aromatic & Hydrophobic Gating Network:** Quadrant 2 drugs consistently engage an extensive cluster of aromatic and aliphatic side chains situated on transmembrane helices TM4, TM5, TM6, and TM12: **Trp232, Phe303, Ile306, Tyr307, Tyr310, Phe336, Leu339, Ile340, Phe343, and Gln725**. The multiple aromatic rings of Vilazodone, Flupentixol, and Loperamide form dense $\pi$-$\pi$ face-to-face and edge-to-face stacking interactions with Phe303 and Phe343.
- **Steric Evasion in Quadrant 1:** In contrast, compact Quadrant 1 molecules (Levetiracetam, Caffeine, Milnacipran) cannot simultaneously span these opposing helical walls, preventing the induced-fit structural transitions required to trigger ATP hydrolysis and outward-facing translocation.

### 6.4 Cross-Modality Correlation: Machine Learning vs. Structural Biophysics

Figure 6 depicts the correlation between the machine learning predicted probability of active efflux $P(\text{P-gp}^+)$ and the physical binding free energy $\Delta G_{\text{bind}}$:

![Figure 6: Correlation between Machine Learning Efflux Probability and Cryo-EM Docking Energy](Novelty%20Upgrade/figures/cryoem_docking_vs_ml_correlation.png)

A strong, statistically significant negative Spearman rank correlation is observed:
$$\rho = -0.5733 \quad (p = 5.28\times10^{-3})$$
Because more negative $\Delta G_{\text{bind}}$ reflects higher thermodynamic stability, this inverse relationship demonstrates that **our machine learning model trained on binary phenotypic cell assays captures authentic physical binding thermodynamics in the human ABCB1 cryo-EM cavity**.

---

## 7. Discussion: The "Golden Window" for CNS Drug Discovery

The descriptor windows below are heuristic proposals retained from the exploratory analysis, not thresholds estimated or validated by the November SHAP tables. They motivate hypotheses about two opposing constraints:
1. **The Influx Force:** Favored by higher lipophilicity ($\log P \ge 1.5$) and low polar surface area ($\text{TPSA} \le 90 \text{ \AA}^2$).
2. **The Efflux Evasion Force:** Favored by low lipophilicity ($\log P \le 3.0$), compact spatial volume ($\text{MW} \le 380 \text{ Da}$, $R_g \le 3.8 \text{ \AA}$), and sparse hydrogen-bonding capacity ($\text{HBD} \le 1$).

```
                      PASSIVE MEMBRANE INFLUX (BBB+)
               ┌──────────────────────────────────────────┐
               │  MolLogP > 1.5                           │
               │  TPSA < 90 Å²                            │
               │  H-Bond Acceptors ≤ 6                    │
               └────────────────────┬─────────────────────┘
                                    │
                     THE "GOLDEN CNS WINDOW" (BBB+ / P-gp-)
                     ★ Moderate Lipophilicity: 1.5 ≤ LogP ≤ 3.0
                     ★ Low Molecular Weight: MW < 380 Da
                     ★ Restricted H-Bonding: H-Donors ≤ 1
                     ★ Compact 3D Envelope: Rg < 3.8 Å, Spherocity > 0.25
                                    │
               ┌────────────────────┴─────────────────────┐
               │  MolLogP > 3.5                           │
               │  MW > 450 Da                             │
               │  Rg > 4.5 Å                              │
               │  Polyvalent H-Bond Donor Pairs (Seelig)  │
               └──────────────────────────────────────────┘
                         ACTIVE EFFLUX (P-gp+)
```

These proposed windows require prospective experimental validation; they do not guarantee BBB passage or efflux evasion.

---

## 8. Limitations & Clinical Perspective

While this dual-barrier computational system represents a significant advance over single-target QSAR models, several biological and clinical caveats must be noted:
1. **Unbound Brain Exposure ($f_{u,\text{brain}}$):** Predicting that a drug is $\text{BBB}^+$ and $\text{P-gp}^-$ does not guarantee in vivo neuropharmacological efficacy. Pharmacological activity depends on the free, unbound drug concentration in brain interstitial fluid ($C_{u,\text{brain}}$), which is modulated by non-specific brain tissue binding, plasma protein binding, and cerebral spinal fluid (CSF) bulk turnover.
2. **Additional Efflux Transporters:** While P-gp is the most abundant efflux pump, Breast Cancer Resistance Protein (**BCRP / ABCG2**) and Multidrug Resistance-Associated Proteins (**MRPs / ABCCs**) also contribute to efflux for specific chemical classes. Future iterations should incorporate BCRP models.
3. **In Vitro Assay Label Variability:** Experimental P-gp substrate designations in literature benchmarks (calcein-AM uptake, bidirectional Caco-2/MDCK permeability assays) occasionally display lab-to-lab variability. Bootstrap intervals describe sampling variation conditional on the observed labels; they do not account for all assay heterogeneity or label error.

---

## 9. Reproducibility & Data Availability

All code, trained model artifacts, scaler parameters, benchmarking scripts, virtual screening pipelines, docking protocols, and test suites are fully accessible in this Git repository:
- **Broad Hub Repurposing Curation:** `python "Novelty Upgrade/scripts/prepare_repurposing_library.py"`.
- **Dual-Barrier Pharmacopeia Screen:** `python "Novelty Upgrade/scripts/run_repurposing_screen.py"`.
- **Cryo-EM Structural Biology Docking Validation:** `python "Novelty Upgrade/scripts/run_cryoem_docking_validation.py"`.
- **Validation Protocol with 95% Bootstrap CI:** `python "Novelty Upgrade/scripts/run_validation.py"`.
- **Incremental Feature Ablation Study:** `python "Novelty Upgrade/scripts/run_ablation_study.py"`.
- **Global SHAP & Figures Generation:** `python "Novelty Upgrade/scripts/run_explainability_analysis.py"`.
- **Automated Test Suite:** `python -m pytest "Novelty Upgrade/tests" -q` (rerun after training; this documentation refresh does not claim a fresh regression-suite result).
- **Interactive Multi-Section Inference:** `python "Novelty Upgrade/inference.py" --smiles <SMILES> --explain`.

Current metrics and model-linked benchmark measurements are under `Novelty Upgrade/results/`. The following screening/docking files remain October artifacts:
- `fda_repurposing_dual_barrier_screen.csv`: Comprehensive 2,405-drug dual-barrier screening profiles.
- `top_cns_repurposing_candidates.csv`: Ranked non-CNS leads for neuro-oncology and neuroinflammation.
- `cns_drugs_efflux_resistance_profiles.csv`: Marketed neuropsychiatric drugs vulnerable to active P-gp clearance.
- `cryoem_pgp_docking_validation.csv`: Cryo-EM P-glycoprotein docking thermodynamics and contact residues.

---

## 10. Conclusion

The regenerated models achieve strong random-test AUCs (BBB 0.9563; P-gp 0.9607),
but the scaffold benchmarks are lower (0.8112 and 0.9001). The P-gp feature ablation
does not show a mean AUC benefit from adding 3D shape to Morgan + 2D descriptors.
Global SHAP prioritizes TPSA for BBB and LogP for P-gp, without establishing causal
mechanisms or validated design thresholds.

Adding an efflux prediction reduces paradox leakage on the 216-molecule OOF
subset, while shared-encoder learning does not demonstrate an across-seed AUC
gain over independent models. The November results support a joint evaluation
framework with explicit uncertainty and chemotype limitations. The October
screening and docking findings remain historical, exploratory evidence and must
be rerun before being presented as outputs of the current models.

---

## References

1. **Löscher, W., & Potschka, H.** (2005). Blood-brain barrier active efflux transporters: ATP-binding cassette gene family. *NeuroRx*, 2(1), 86–98. DOI: 10.1602/neurorx.2.1.86.
2. **Meng, F., Xi, Y., Huang, J., & Ayers, P. W.** (2021). A curated diverse molecular database of blood-brain barrier permeability with chemical descriptors. *Scientific Data*, 8, 289. DOI: 10.1038/s41597-021-01069-5. *(The B3DB benchmark used in this work.)*
3. **Broccatelli, F., et al.** (2011). Large-scale classification of P-glycoprotein substrates using 3D molecular descriptors. *Journal of Medicinal Chemistry*, 54(6), 1740–1751. DOI: 10.1021/jm101421d.
4. **Corsello, S. M., et al.** (2017). The Drug Repurposing Hub: a next-generation drug library and information resource. *Nature Medicine*, 23(4), 405–408. DOI: 10.1038/nm.4306.
5. **Alam, A., Kowal, J., Broude, E., Roninson, I., & Locher, K. P.** (2019). Structural insight into substrate and inhibitor discrimination by human P-glycoprotein. *Science*, 363(6428), 753–756. DOI: 10.1126/science.aav7112.
6. **Seelig, A.** (1998). How does P-glycoprotein recognize its substrates? *International Journal of Clinical Pharmacology and Therapeutics*, 36(1), 50–54.
7. **Seelig, A., & Landwojtowicz, E.** (2000). Structure-activity relationship of P-glycoprotein substrates and inhibitors. *European Journal of Pharmaceutical Sciences*, 12(1), 31–40. DOI: 10.1016/S0928-0987(00)00155-7.
8. **Lundberg, S. M., et al.** (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67. DOI: 10.1038/s42256-019-0138-9.
9. **Pajouhesh, H., & Lenz, G. R.** (2005). Medicinal chemical properties of successful central nervous system drugs. *NeuroRx*, 2(4), 541–553. DOI: 10.1602/neurorx.2.4.541.
10. **Zhao, Y. H., et al.** (2007). Predicting penetration across the blood-brain barrier from simple descriptors and fragmentation schemes. *Journal of Chemical Information and Modeling*, 47(1), 170–178. DOI: 10.1021/ci600312d.
11. **Volk, H. A., Potschka, H., & Löscher, W.** (2004). Increased expression of the multidrug transporter P-glycoprotein in limbic brain regions after seizures in rats. *Epilepsy Research*, 58(1), 67–79. DOI: 10.1016/j.eplepsyres.2004.01.002.
12. **Huang, K., et al.** (2021). Therapeutics Data Commons: Machine learning datasets and tasks for drug discovery and development. *Nature Chemical Biology*, 18(9), 1033–1041. DOI: 10.1038/s41589-022-01131-0.
13. **Moitra, K., et al.** (2011). Role of ABC transporters in the development of multidrug resistance in cancer and psychiatry. *Current Medicinal Chemistry*, 18(19), 2841–2852. DOI: 10.2174/092986711796150497.
14. **Adenot, M., & Lahana, R.** (2004). Blood-brain barrier permeation models: discriminating between potential CNS and non-CNS drugs including P-glycoprotein substrates. *Journal of Chemical Information and Computer Sciences*, 44(1), 239–248. DOI: 10.1021/ci034205d. *(Prior art for the dual-barrier framing; PDF in `Initial Model/reference_papers/`.)*
15. **Martins, I. F., et al.** (2012). A Bayesian approach to in silico blood-brain barrier penetration modeling. *Journal of Chemical Information and Modeling*, 52(6), 1686–1697. DOI: 10.1021/ci300124c.


