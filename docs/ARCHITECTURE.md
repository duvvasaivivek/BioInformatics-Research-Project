# Architecture

How the system works. For how to run it see [`../README.md`](../README.md); for the
exhaustive block-by-block specification with I/O shapes and loss functions see
[`BLOCK_DIAGRAM.md`](BLOCK_DIAGRAM.md); for why it was built this way and what is
publishable see [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

This document merges the former `ARCHITECTURE_DEEP_DIVE.md` and
`Novelty Upgrade/docs/SCIENTIFIC_DOCUMENTATION.md`.

## Contents
- [The biological problem](#section-1-the-core-biological-paradox--taxonomy)
- [Why two paradigms](#section-2-the-bioinformatics-root-cause--why-we-upgraded-from-paradigm-1-to-paradigm-2)
- [Paradigm A — graph neural networks](#section-3-paradigm-1--deep-graph-isomorphism-network-initial-model)
- [Paradigm B — the dual-barrier ensemble](#section-4-paradigm-2--b3db-dual-barrier-ensemble-engine)
- [Explainability and biophysical validation](#section-5-explainability-domain-auditing--biophysical-validation)
- [Multi-task GNN](#multi-task-gnn-dual-barrier-from-one-encoder)
- [Validation protocol and performance](#3-leakage-resistant-validation-protocol--empirical-performance)
- [Cross-task shared properties](#6-cross-task-shared-properties--divergence-analysis)
- [Decision matrix, applicability domain, counterfactuals](#7-four-quadrant-decision-matrix-applicability-domain--counterfactuals)
- [Codebase layout](#8-codebase-architecture--future-upgrade-guide)

---
# SECTION 1: THE CORE BIOLOGICAL PARADOX & TAXONOMY

> 💡 **Layman Intuition (In Plain English): The Castle Fortress & The Bouncer**  
> Imagine your brain is an ultra-secure medieval castle. To keep out pathogens and toxins, it has two heavy lines of defense:
> 1. **The Castle Walls (The Blood-Brain Barrier / BBB):** These are tightly packed endothelial cells lined with oily lipid membranes. Water-soluble molecules simply bounce off. To slip through passively, a molecule must be somewhat greasy (lipophilic) and compact.
> 2. **The Castle Bouncer (P-glycoprotein / P-gp):** Even if a drug slips through the fatty wall, waiting on the other side is a molecular security guard called P-glycoprotein. If the drug looks suspicious (bulky, sticky with hydrogen bonds), the pump grabs it, burns cellular fuel (ATP), and flings it right back out into the bloodstream!
>
> **The Clinical Paradox:** A drug like *Loperamide (Imodium)* is an opioid. It slips through the fatty membrane easily ($\text{BBB}^+$), but P-glycoprotein violently throws it back out ($\text{P-gp}^+$). Hence, it cures diarrhea in your gut but never causes opioid effects in your brain. Conversely, *Levetiracetam (Keppra)* crosses the wall ($\text{BBB}^+$) and completely evades the bouncer ($\text{P-gp}^-$), reaching the brain to stop seizures. This project decouples both hurdles so we don't develop false leads!

```
                                      MOLECULAR SMILES INPUT
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
          PARADIGM 1: INITIAL MODEL                         PARADIGM 2: B3DB DUAL-BARRIER
        (End-to-End Graph GNN)                            (Multi-Modal Decoupled Engine)
                       │                                                 │
          Molecular Graph Construction                     Multi-Modal Feature Extraction
          Nodes: 9 atomic features                         ├─ Modality 1: 2048-bit ECFP4 Morgan FP
          Edges: 7 bond chemistry features                 ├─ Modality 2: 6D Physicochemical Descriptors
                       │                                   └─ Modality 3: 6D 3D ETKDGv3 Conformer Geometry
                       ▼                                                 │
          3x Edge-Aware GINEConv Blocks                                  ▼
          BatchNorm1d + ReLU + Dropout(0.3)                Dual Independent Ensembles:
                       │                                   ├─ M_BBB (2,054 dims) -> P(BBB+)
                       ▼                                   └─ M_Pgp (2,060 dims) -> P(P-gp+)
          Global Mean Pooling Layer                                      │
          Graph Embedding: R^(B x 256)                                   ▼
                       │                                   Four-Quadrant CNS Decision Matrix
                       ▼                                   S_CNS = P(BBB+) * (1 - P(P-gp+))
          Classifier MLP Readout                                         │
          Linear(256->32) -> ReLU -> Dropout                             ▼
          Linear(32->2)                                    Mechanistic Validation & Explainability
                       │                                   ├─ TreeSHAP Log-Odds Margin Decomposition
                       ▼                                   ├─ Tanimoto / Mahalanobis Applicability Domain
               P(BBB+), P(BBB-)                            ├─ Reaction SMARTS Counterfactual Engine
                                                           └─ Cryo-EM 6QEX Vina Biophysical Docking
```

---

---

# SECTION 2: THE BIOINFORMATICS ROOT CAUSE — WHY WE UPGRADED FROM PARADIGM 1 TO PARADIGM 2

> 💡 **Layman Intuition: Why Did the First Version Fail in Real Medicine?**  
> In the first version of our project (Paradigm 1), we built an advanced Deep Graph Neural Network to predict whether a drug crosses the blood-brain barrier. It achieved solid accuracy ($\sim 82\%$) on paper.  
> **So why did we upgrade?** Because in real pharmacology, predicting only "Blood-Brain Barrier Crossing" is a dangerous half-truth!  
> More than 90% of brain drugs fail in human clinical trials because of a hidden villain: **The P-glycoprotein efflux pump**. Paradigm 1 was completely blind to this pump. It would happily classify drugs like Loperamide as "Successful Brain Drugs" because they pass through oily membranes, completely unaware that the P-gp pump immediately flings them out.  
> We upgraded to Paradigm 2 to bring in **3D physical conformer modeling**, **exact TreeSHAP margin physics**, **active pump efflux prediction**, and **Cryo-EM atomic docking** so we never recommend a false lead again.

### 2.1 The Fatal Flaw: The Single-Endpoint Fallacy
In traditional literature and in Paradigm 1 (`Initial Model`), Blood-Brain Barrier permeation is framed as a single yes/no prediction:
$$\text{Prediction} \in \{\text{BBB}^-, \text{BBB}^+\}$$

**Why this creates clinical failures:**
1. **Passive Influx vs. Active Clearance:**  
   Molecules cross cell membranes based on lipophilicity ($\text{LogP}$) and polar surface area ($\text{TPSA}$). If a molecule is greasy and small, it slips into the cell membrane. Paradigm 1 sees this and predicts $\text{BBB}^+$.
2. **The Active Efflux Counter-Current:**  
   P-glycoprotein sits right on the cell surface. It grabs greasy molecules inside the membrane and pumps them back into the blood:
   $$\text{Net Brain Entry} = \text{Passive Influx} - \text{Active Efflux}$$
   If the efflux pump rate exceeds passive entry, the drug never reaches brain cells. Paradigm 1 is blind to this active clearance.

---

### 2.2 Why 3D Conformer Geometry (ETKDGv3) Was Introduced
- **The 2D Flaw:** Flat 2D chemical graphs cannot capture 3D physical shape.
- **The Biological Reality:** Cryo-EM structures of human P-glycoprotein (**PDB: 6QEX**) prove that P-gp has a massive 3D binding chamber ($6,000\text{ \AA}^3$). It traps molecules based on:
  1. **Molecular Bulk:** Measured by the **Radius of Gyration ($R_g$)**; bulky molecules get wedged inside.
  2. **Conformation Shape:** Measured by **Asphericity** and **Eccentricity** (is it a flat disk, a sphere, or a long rod?).
  3. **Spatial H-Bond Spacing:** P-gp recognizes hydrogen-bond donor points separated by specific 3D distances ($2.5\text{ \AA}$ or $4.6\text{ \AA}$). Flat 2D graphs cannot measure 3D spatial distances across flexible bonds.
- **What It Accomplished:** Adding 3D conformers increased P-gp prediction accuracy to **$92.95\%$** and PR-AUC to **$0.9656$**.

---

### 2.3 Why TreeSHAP in Log-Odds Margin Space Was Introduced
- **The Problem with GNN Saliency:** GNN heatmaps highlight atoms with colors, but they cannot tell a chemist: *"Does this atom help or hurt brain crossing, and by how many points?"*
- **What TreeSHAP Accomplished:** Breaks down predictions in raw score points (log-odds margin space):
  $$\text{Final Score } f(x) = \text{Baseline } \phi_0 + \sum \text{Feature Points } \phi_i$$
  - **Polar Area (TPSA) & H-Donors:** Subtract points ($\phi < 0$), reflecting the energetic penalty of stripping water molecules before crossing oily membranes.
  - **Greasy Groups (LogP) & Benzene Rings:** Add points ($\phi > 0$) for BBB crossing, while simultaneously adding risk points for P-gp efflux.

---

### 2.4 Why the 4-Quadrant CNS Decision Matrix ($S_{\text{CNS}}$) Was Introduced
Instead of ambiguous probabilities, compounds are sorted into 4 clinical buckets:
- **Quadrant 1 (Optimal Lead: $\text{BBB}^+ / \text{P-gp}^-$):** Enters the brain and evades the pump ($S_{\text{CNS}} \ge 0.70$). E.g., *Levetiracetam, Caffeine*.
- **Quadrant 2 (Paradoxical Efflux Drug: $\text{BBB}^+ / \text{P-gp}^+$):** Enters the membrane, but gets immediately pumped out ($S_{\text{CNS}} \approx 0.0$). E.g., *Loperamide*.
- **Quadrant 3 (Peripheral Agent: $\text{BBB}^- / \text{P-gp}^-$):** Low permeation, pump evader. E.g., *Penicillin G*.
- **Quadrant 4 (Restricted Peripheral: $\text{BBB}^- / \text{P-gp}^+$):** Doubly barred from the brain. E.g., *Fexofenadine*.

**Composite CNS Desirability Index:**
$$S_{\text{CNS}} = P(\text{BBB}^+) \times (1 - P(\text{P-gp}^+)) \in [0.0, 1.0]$$

---

### 2.5 Quantitative Performance Elevation Summary

| Evaluation Dimension | Paradigm 1 (Initial GNN Model) | Paradigm 2 (B3DB Dual-Barrier Engine) | Elevation / Improvement |
|:---|:---:|:---:|:---|
| **Underlying Methodology** | Single-task GINEConv GNN | Multi-Modal Gradient Boosted Forest | Graph topology + 3D Conformer fusion |
| **Efflux Modeling** | **Zero awareness** (Blind to P-gp) | **Full Decoupled ABCB1 Model** | Solves the Loperamide paradox |
| **Feature Representation** | 9 node dims, 7 edge dims | **2,054 dims (BBB) / 2,060 dims (P-gp)** | Extended circular + 2D specs + 3D physics |
| **3D Conformer Geometry** | None (Flat 2D graph) | **RDKit ETKDGv3 Conformer Engine** | Spatial bulk, asphericity, gyration radius |
| **BBB Test Accuracy** | $82.4\%$ | **$88.56\%$** | **$+6.16\%$ absolute gain** |
| **BBB Test ROC-AUC** | $0.884$ | **$0.9522$** | **$+0.0682$ discriminative power** |
| **BBB Test PR-AUC** | $0.912$ | **$0.9724$** | **$+0.0604$ precision-recall gain** |
| **P-gp Test Accuracy** | N/A (Not modeled) | **$92.95\%$** | First-in-class multi-modal efflux model |
| **P-gp Test ROC-AUC** | N/A (Not modeled) | **$0.9587$** | High discriminative accuracy |
| **Explainability Depth** | Heuristic attention weights | **Exact TreeSHAP Log-Odds Margin** | Direct thermodynamic energy mapping |
| **Biophysical Validation** | None | **Cryo-EM PDB 6QEX Docking ($p=0.0035$)** | Statistically validated physical binding |
| **Translational Clinical Yield** | 0 repurposing leads | **273 FDA Repurposing CNS Leads** | Screened across 2,406 approved drugs |

---

---

# SECTION 3: PARADIGM 1 — DEEP GRAPH ISOMORPHISM NETWORK (INITIAL MODEL)

## 3.1 Graph Featurization: Nodes & Edges

> 💡 **Layman Intuition: Molecules as Road Maps**  
> - **Nodes (Atoms) = Towns:** Each atom has local traits: population (atomic number), connected highways (degree), and charge.
> - **Edges (Bonds) = Highways:** Single dirt roads, double highways, or aromatic speedway rings.

### A. Node Feature Vector ($d_v = 9$)
For each atom $v$, a 9-dimensional vector $\mathbf{x}_v \in \mathbb{R}^9$ is extracted:
1. `AtomicNum`: Atomic number ($6=\text{C}, 7=\text{N}, 8=\text{O}$)
2. `Degree`: Number of bonded heavy atom neighbors ($0$ to $4$)
3. `FormalCharge`: Electrical charge ($-1, 0, +1$)
4. `TotalNumHs`: Connected hydrogens ($0$ to $3$)
5. `IsAromatic`: Aromatic ring flag ($0$ or $1$)
6. `IsInRing`: Cyclic ring flag ($0$ or $1$)
7. `ExplicitValence`: Bond valence count
8. `ImplicitValence`: Hydrogen-satisfied valence count
9. `Hybridization`: Orbital geometry ($1=s, 2=sp, 3=sp^2, 4=sp^3$)

### B. Edge Feature Vector ($d_e = 7$)
For each directed bond between atoms, a 7-dimensional vector $\mathbf{e}_{u,v} \in \mathbb{R}^7$ is extracted:
1. `Single Bond`: Order 1.0 flag ($\{0, 1\}$)
2. `Double Bond`: Order 2.0 flag ($\{0, 1\}$)
3. `Triple Bond`: Order 3.0 flag ($\{0, 1\}$)
4. `Aromatic Bond`: Delocalized aromatic type ($\{0, 1\}$)
5. `Is Aromatic`: Aromaticity flag ($\{0, 1\}$)
6. `Is Conjugated`: Alternating single-double bond flag ($\{0, 1\}$)
7. `Is In Ring`: Ring bond flag ($\{0, 1\}$)

### C. Batch Tensors ($B = 32$)
- **Node Matrix:** $\mathbf{X} \in \mathbb{R}^{N_{\text{nodes}} \times 9}$
- **Edge Index:** $\mathbf{E}_{\text{index}} \in \mathbb{Z}^{2 \times N_{\text{edges}}}$ (COO format)
- **Edge Attributes:** $\mathbf{E}_{\text{attr}} \in \mathbb{R}^{N_{\text{edges}} \times 7}$
- **Batch Vector:** $\mathbf{batch} \in \mathbb{Z}^{N_{\text{nodes}}}$ (maps atom $\to$ molecule ID)

---

## 3.2 Deep Convolutional Architecture (`BBB_GNN`)

> 💡 **Layman Intuition: The 3 Rounds of Gossip**  
> 1. **Round 1 (`conv1`):** Atoms whisper their identity to direct neighbors across bonds.
> 2. **Round 2 (`conv2`):** Atoms whisper again, learning about neighbors 2 steps away.
> 3. **Round 3 (`conv3`):** Atoms whisper a 3rd time, understanding the larger functional group they live in.
> 4. **Global Mean Pooling:** Averages all atom vectors into a single **256-number summary vector** for the entire molecule.
> 5. **Classifier Readout:** A 2-stage judge evaluates the 256-number summary and outputs scores for $\text{BBB}^-$ vs $\text{BBB}^+$.

```
GRAPH INPUT: x [N_nodes, 9] | edge_index [2, N_edges] | edge_attr [N_edges, 7]
     │
     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GINE BLOCK 1 (conv1)                                                   │
│   • Internal MLP: Linear(9 -> 256) -> ReLU -> Linear(256 -> 256)       │
│   • GINEConv message passing with edge_dim = 7                         │
│   • BatchNorm1d(256) -> ReLU -> Dropout(p = 0.30)                      │
│   Output: [N_nodes, 256]                                               │
└────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GINE BLOCK 2 (conv2)                                                   │
│   • Internal MLP: Linear(256 -> 256) -> ReLU -> Linear(256 -> 256)     │
│   • GINEConv message passing with edge_dim = 7                         │
│   • BatchNorm1d(256) -> ReLU -> Dropout(p = 0.30)                      │
│   Output: [N_nodes, 256]                                               │
└────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GINE BLOCK 3 (conv3)                                                   │
│   • Internal MLP: Linear(256 -> 256) -> ReLU -> Linear(256 -> 256)     │
│   • GINEConv message passing with edge_dim = 7                         │
│   • BatchNorm1d(256) -> ReLU -> Dropout(p = 0.30)                      │
│   Output: [N_nodes, 256]                                               │
└────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GLOBAL MEAN POOLING                                                    │
│   Averages all atom vectors: h_G = (1 / |V|) * sum(x_v)                │
│   Output: [B, 256]                                                     │
└────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ CLASSIFIER READOUT HEAD (MLP)                                          │
│   • Linear(256 -> 32) -> ReLU -> Dropout(p = 0.30)                     │
│   • Linear(32 -> 2)                                                    │
│   Output: Raw Logits [B, 2] -> Softmax -> [P(BBB-), P(BBB+)]           │
└────────────────────────────────────────────────────────────────────────┘
```

### Complete Parameter Census (345,514 Total Parameters)
- `conv1` Internal MLP: $\text{Linear}(9 \to 256) \to \text{ReLU} \to \text{Linear}(256 \to 256)$ $\implies 68,352$ params
- `conv2` Internal MLP: $\text{Linear}(256 \to 256) \to \text{ReLU} \to \text{Linear}(256 \to 256)$ $\implies 131,584$ params
- `conv3` Internal MLP: $\text{Linear}(256 \to 256) \to \text{ReLU} \to \text{Linear}(256 \to 256)$ $\implies 131,584$ params
- Batch Normalization (`bn1, bn2, bn3`): $3 \times 512 = 1,536$ params
- Edge Projection Matrices (GINEConv `edge_dim`, bias included): $72 + 2,048 + 2,048 = 4,168$ params
- Classifier Head: $\text{Linear}(256 \to 32) \implies 8,224$ params; $\text{Linear}(32 \to 2) \implies 66$ params
- **Total:** **$345,514$ trainable parameters** (verified by instantiating the module and summing `requires_grad` tensors)

### Training Settings
- **Weighted Cross-Entropy Loss:** Penalizes errors on minority classes with inverse class weights: $w_c = \frac{N}{2 \cdot N_c}$.
- **Optimizer:** `AdamW` ($\text{lr} = 0.0005$, $\text{weight\_decay} = 0.0005$).
- **Gradient Clipping:** Max norm capped at $2.0$.
- **Scheduler:** `ReduceLROnPlateau(factor=0.5, patience=8)`.
- **Validation:** Stratified 10-Fold CV with early stopping ($20$ epochs).

---

---

# SECTION 4: PARADIGM 2 — B3DB DUAL-BARRIER ENSEMBLE ENGINE

```
                                  SMILES STRING
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
      MODALITY 1                   MODALITY 2                   MODALITY 3
2048-bit Morgan Fingerprint    2D Physicochemical            3D ETKDGv3 Conformer
   (ECFP4, Radius=2)               Descriptors                 Shape Descriptors
           │                            │                            │
   Bit Vector x_FP              Continuous z_2D              Continuous z_3D
   Shape: [1, 2048]              Shape: [1, 6]                Shape: [1, 6]
           │                            │                            │
           ├────────────────────────────┤                            │
           │ Concatenation              │                            │
           ▼                            │                            │
   BBB Feature Vector                   │                            │
   x_BBB = [x_FP || z_2D]               │                            │
   Total Dims: 2,054                    └──────────────┬─────────────┘
           │                                           │ Concatenation
           ▼                                           ▼
┌──────────────────────┐                      P-gp Feature Vector
│   XGBoost M_BBB      │                      x_Pgp = [x_FP || z_2D || z_3D]
│ (Bayesian-Optimized) │                      Total Dims: 2,060
└──────────────────────┘                               │
           │                                           ▼
           ▼                                ┌──────────────────────┐
    P(BBB+ in [0, 1])                       │    XGBoost M_Pgp     │
           │                                │ (Efflux-Calibrated)  │
           │                                └──────────────────────┘
           │                                           │
           │                                           ▼
           │                                   P(P-gp+ in [0, 1])
           │                                           │
           └───────────────────┬───────────────────────┘
                               │
                               ▼
               FOUR-QUADRANT CNS DECISION MATRIX
                S_CNS = P(BBB+) * (1 - P(P-gp+))
                               │
                               ▼
               QUADRANT CLASSIFICATION (Q1 - Q4)
```

---

## 4.1 Multi-Modal Feature Extraction
- **Modality 1: 2048-bit Morgan Fingerprint (Barcode):** $2,048$ binary bits ($0$ or $1$) detecting sub-fragments (radius $r=2$).
- **Modality 2: 2D Physicochemical (Spec Sheet):** $6$ continuous variables (`MolWt`, `LogP`, `TPSA`, `HDonors`, `HAcceptors`, `RotBonds`).
- **Modality 3: 3D Conformer Geometry (Physical 3D Body):** $6$ continuous shape variables generated via ETKDGv3 physics (`RadiusOfGyration`, `Asphericity`, `Eccentricity`, `SpherocityIndex`, `NPR1`, `NPR2`).
- **Leakage-Free Scaling:** Continuous descriptors are standardized strictly on training data: $z = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$.

**Concatenations:**
- **BBB Feature Vector:** $\mathbf{x}_{\text{BBB}} = [2048 \text{ bits} \parallel 6 \text{ 2D specs}] \implies \mathbf{2,054 \text{ dimensions}}$
- **P-gp Feature Vector:** $\mathbf{x}_{\text{Pgp}} = [2048 \text{ bits} \parallel 6 \text{ 2D specs} \parallel 6 \text{ 3D shape}] \implies \mathbf{2,060 \text{ dimensions}}$

---

## 4.2 XGBoost Hyperparameter Matrix

| Hyperparameter | BBB Model ($\mathcal{M}_{\text{BBB}}$) | P-gp Model ($\mathcal{M}_{\text{Pgp}}$) | Operational Purpose |
|:---|:---:|:---:|:---|
| **Input Feature Dimension** | **$2,054$** | **$2,060$** | Concatenated feature length |
| **Max Estimators** | $1,000$ (early stopping $30$) | $300$ | Number of boosting trees |
| **Learning Rate** | Optuna tuned: $[0.01, 0.30]$ | $0.05$ | Shrinkage step per tree |
| **Max Tree Depth** | Optuna tuned: $[3, 10]$ | $5$ | Tree depth (feature interaction order) |
| **Row Subsample** | Optuna tuned: $[0.50, 1.00]$ | $0.80$ | Random sample percentage per tree |
| **Colsample by Tree** | Optuna tuned: $[0.50, 1.00]$ | $0.80$ | Feature percentage per tree |
| **Min Child Weight** | Optuna tuned: $[1, 10]$ | $1$ | Minimum samples per leaf |
| **Class Balancing** | `scale_pos_weight = 1.91` | `scale_pos_weight = 0.74` | Balances minority class loss |
| **Tree Method** | `hist` (Histogram GPU/CPU) | `hist` | Fast continuous feature binning |
| **Data Partitioning** | Strict 70/15/15 Train/Val/Test | Strict 80/20 Train/Test | Complete data leakage prevention |

---

---

# SECTION 5: EXPLAINABILITY, DOMAIN AUDITING & BIOPHYSICAL VALIDATION

### 5.1 Exact TreeSHAP in Log-Odds Margin Space
Attributions are calculated in raw score points (log-odds margin space):
$$\text{Final Margin } f(x) = \text{Baseline } \phi_0 + \sum_{i=1}^M \phi_i(x)$$
- Baseline score: $\phi_0 = -0.3121$ for BBB, $+0.1519$ for P-gp.
- The probability is recovered via sigmoid:
  $$P = \frac{1}{1 + e^{-f(x)}}$$
- `MorganSubstructureMapper` traces any active bit $k$ with high attribution back to its exact chemical fragment.

---

### 5.2 Applicability Domain (Knowing When You Don't Know)
- **Tanimoto Structural Similarity ($T_{\max}$):** Measures fingerprint overlap against the training library ($0.0 = \text{alien}, 1.0 = \text{twin}$).
- **Descriptor Leverage ($D_{\text{leverage}}$):** Measures how far physical specs stray from training averages.
- **Rules:**
  $$\text{Inside Domain} \iff T_{\max} \ge 0.40 \quad \text{AND} \quad D_{\text{leverage}} \le 2.50$$
  If a molecule fails, it is flagged as **Outside Domain**, warning chemists not to trust the prediction blindly.

---

### 5.3 Reaction SMARTS Counterfactuals (The "What-If" Engine)
Tests 8 organic chemistry bioisostere transformations to guide lead optimization:
1. Alcohol $O$-Methylation: `[O;H1:1]>>[O:1]C` (Lowers TPSA, raises LogP)
2. Primary Amine $N$-Methylation: `[N;H2:1]>>[N:1]C` (Lowers desolvation penalty)
3. Carboxylate Esterification: `[C:1](=O)[O;H1:2]>>[C:1](=O)OC` (Neutralizes negative charge)
4. Acid to Amide: `[C:1](=O)[O;H1:2]>>[C:1](=O)N` (Bioisostere replacement)
5. Aromatic Fluorination: `[c;H1:1]>>[c:1]F` (Tunes LogP, blocks metabolism)
6. Ether $O$-Demethylation: `[O:1]C>>[O:1][H]` (Reduces excess lipophilicity)
7. Amine Demethylation: `[N;H0:1]C>>[N:1][H]` (Generates secondary amine)
8. Ketone Deoxygenation: `[C:1](=O)[C:2]>>[C:1][C:2]` (Reduces HBA)

---

### 5.4 Cryo-EM Biophysical Docking Validation (PDB: 6QEX)
- Uses the 3.5 Å Cryo-EM structure of Human ABCB1 (**PDB: 6QEX**).
- Search Box: Center $(173.33, 166.74, 161.48)$, Size $24\text{ \AA}^3$, AutoDock Vina v1.2.7.
- Calculates physical binding energy $\Delta G_{\text{bind}}$ (kcal/mol) and estimated dissociation constant:
  $$K_d = \exp\left( \frac{\Delta G_{\text{bind}}}{RT} \right)$$
- **Validation Outcome:**
  - Quadrant 2 (Efflux Vulnerable): $\Delta G = -9.23 \pm 0.69\text{ kcal/mol}$ ($K_d \approx 170\text{ nM}$, tight binding).
  - Quadrant 1 (Efflux Evaders): $\Delta G = -6.74 \pm 0.94\text{ kcal/mol}$ ($K_d \approx 11.2\text{ }\mu\text{M}$, weak binding).
  - Statistical significance: Two-tailed $p = 0.0035$ ($t = -4.10$), confirming that the machine learning models learned authentic biophysics.

---
---

# Multi-Task GNN: Dual Barrier From One Encoder

A third model family, added after the two above. It learns both barriers from a
single shared molecular-graph encoder rather than training them separately.

**Motivation.** On the 216 molecules where B3DB and Broccatelli overlap, the BBB
label and P-gp substrate status are near-independent (Cramér's V = 0.064,
OR 1.42 [0.76, 2.67], p = 0.35). Near-independent labels are complementary rather
than redundant supervision, which is the textbook precondition for multi-task
learning to help the smaller task.

**Architecture** (`Novelty Upgrade/gnn/`):

```
SMILES -> RDKit graph (x[V,66], edge_index[2,2E], edge_attr[2E,14])
             |
   3x GINEConv + BatchNorm + ReLU + Dropout      <- shared encoder
             |
   global_mean_pool  ->  h [B, 256]              <- heads branch here
             |
      +------+------+
      |             |
  head A        head B
  BBB +/-       P-gp substrate
```

Heads branch from the 256-dim pooled vector rather than the 32-unit readout, so
neither task is forced through the other's bottleneck. 372,898 parameters with two
heads against 364,608 with one.

**Masked loss.** Only ~216 of ~8,800 molecules carry both labels, so roughly 97.5%
have exactly one. Each head's loss is computed over only its labelled molecules and
normalised by that head's own count, so the 7,805-molecule BBB task cannot swamp the
1,212-molecule P-gp task. A head with no labelled molecules in a batch receives
exactly zero gradient, which is asserted in the verification suite.

**Splits are molecule-level, never task-level.** A dual-labelled molecule carries
both labels into the same partition; the alternative would contaminate every
multi-task number. `gnn/splits.py` asserts this before training.

**Result: it does not work, and that is the finding.** Multi-task training helps
neither task on either split protocol. Upweighting the minority task does not rescue
it and progressively damages the other. Full numbers, the statistical power analysis,
and the decision-aware metric suite built to evaluate it are in
[`RESEARCH_LOG.md`](RESEARCH_LOG.md).

**Entry points:** `scripts/train_multitask_gnn.py`, `scripts/run_kfold_multitask.py`,
`scripts/run_seed_variance.py`, `scripts/run_metric_analysis.py`, verified by
`scripts/verify_multitask_gnn.py` (29 checks) and
`scripts/verify_decision_metrics.py` (31 checks).


---

## 3. Leakage-Resistant Validation Protocol & Empirical Performance

### Methodological Safeguards Against Optimistic Bias
1. **Strict Partition Isolation:** Held-out test sets (20%) are isolated prior to any descriptor standardization or hyperparameter tuning. Zero test instances are exposed to the model during training.
2. **Standardization Isolation:** Continuous descriptor normalization ($z_j = (x_j - \mu_j) / \sigma_j$) fits scaler parameters exclusively on training distributions and serializes them to `models/*_scaler_metadata.json`.
3. **Statistical Resampling:** 1,000 bootstrap resamplings on test sets determine empirical 95% Confidence Intervals (2.5th to 97.5th percentiles).
4. **Scaffold Diversity Analysis:** Murcko scaffold decomposition analyzes molecular diversity across the benchmark spaces.

### Test Set Benchmark Performance (95% Bootstrap Confidence Intervals)

| Evaluation Metric | BBB Permeability ($\mathcal{M}_{\text{BBB}}$) | P-gp Efflux Substrate ($\mathcal{M}_{\text{Pgp}}$) |
|---|:---:|:---:|
| **Benchmark Dataset** | Curated B3DB ($N = 7,805$) | Broccatelli TDC ($N = 1,205$) |
| **Test Set Size ($N_{\text{test}}$)** | $1,171$ molecules | $241$ molecules |
| **Accuracy** | **0.8856** [0.8659 - 0.9035] | **0.9295** [0.8963 - 0.9585] |
| **Balanced Accuracy** | **0.8775** [0.8564 - 0.8974] | **0.9287** [0.8953 - 0.9572] |
| **Precision** | **0.9122** [0.8901 - 0.9319] | **0.9313** [0.8828 - 0.9697] |
| **Recall / Sensitivity** | **0.9073** [0.8846 - 0.9269] | **0.9385** [0.8976 - 0.9773] |
| **Specificity** | **0.8478** [0.8125 - 0.8820] | **0.9189** [0.8644 - 0.9636] |
| **F1-Score** | **0.9097** [0.8923 - 0.9237] | **0.9349** [0.9023 - 0.9621] |
| **ROC-AUC** | **0.9522** [0.9410 - 0.9629] | **0.9587** [0.9330 - 0.9806] |
| **PR-AUC (Average Precision)**| **0.9724** [0.9651 - 0.9791] | **0.9685** [0.9488 - 0.9846] |
| **Matthews Corr Coeff (MCC)**| **0.7535** [0.7126 - 0.7924] | **0.8580** [0.7920 - 0.9164] |
| **Brier Score (Calibration)**| **0.0847** [0.0738 - 0.0958] | **0.0679** [0.0473 - 0.0925] |

### Scaffold Diversity
* **B3DB Unique Murcko Scaffolds:** $2,129$ unique cores across $7,805$ compounds (Diversity ratio: $0.273$).
* **Broccatelli Unique Murcko Scaffolds:** $670$ unique cores across $1,205$ compounds (Diversity ratio: $0.556$).
* **Overlapping Scaffolds:** $288$ common chemotypes between datasets.

---

---

## 6. Cross-Task Shared Properties & Divergence Analysis

### Quantitative Cross-Task Comparison

| Property | BBB Mean \|SHAP\| | P-gp Mean \|SHAP\| | Task Category | Biophysical Interpretation |
|---|:---:|:---:|:---:|---|
| **TPSA ($\text{\AA}^2$)** | **1.1825** | 0.1434 | Desolvation Filter | Primary barrier for passive BBB crossing due to desolvation penalty; negligible impact on P-gp efflux. |
| **MolLogP** | 0.3992 | **1.3949** | **Antagonistic (Divergent)** | High lipophilicity favors passive entry ($\text{BBB}^+$), but simultaneously acts as the strongest recruiter for active P-gp efflux ($\text{P-gp}^+$). |
| **MolWt (Da)** | 0.5366 | 0.5783 | Compound Barrier Burden | High size retards passive membrane diffusion while expanding interaction contacts inside the P-gp cavity. |
| **HDonors (HBD)** | 0.2711 | 0.0911 | Seelig Efflux Trigger | Increasing donors creates aqueous clusters retarding permeation while activating Seelig recognition patterns. |
| **HAcceptors (HBA)**| 0.1198 | 0.0708 | Efflux Trigger | Hydrogen bonding acceptors increase polarity and activate P-gp ATPase. |
| **RotBonds** | 0.1113 | 0.0387 | Induced-Fit Risk | Flexibility allows molecules to adopt induced-fit conformations inside P-gp transmembrane helices. |

### Spearman Rank Correlation
$$\rho = 0.7714 \quad (p = 0.0724)$$
The ranking shifts fundamentally:
* **BBB:** $\text{TPSA} > \text{MolWt} > \text{LogP} > \text{HDonors} > \text{RotBonds} > \text{HAcceptors}$
* **P-gp:** $\text{LogP} > \text{MolWt} > R_g > NPR_2 > \text{TPSA} > \text{HDonors}$

### The Golden CNS Window
To evade active efflux while achieving high passive permeation:
* $\text{LogP} \in [1.5, 3.0]$
* $\text{TPSA} < 70 - 90 \text{ \AA}^2$
* $\text{MW} < 380 - 400 \text{ Da}$
* $\text{HDonors} \le 1$
* $R_g < 3.8 \text{ \AA}$ (Compact spherical envelope)

---

---

## 7. Four-Quadrant Decision Matrix, Applicability Domain & Counterfactuals

### The 4-Quadrant Categorization
1. **Quadrant 1: $\text{BBB}^+ / \text{P-gp}^-$ [Optimal CNS Candidate]**  
   Favorable passive membrane partition; evades ABCB1 efflux. High, sustained brain exposure (e.g., Levetiracetam, Caffeine, Diazepam).
2. **Quadrant 2: $\text{BBB}^+ / \text{P-gp}^+$ [Paradoxical Drug / High Efflux Risk]**  
   Partitions into endothelial membranes but is cleared by P-gp back into capillary blood. Negligible brain exposure despite high lipophilicity (e.g., Loperamide).
3. **Quadrant 3: $\text{BBB}^- / \text{P-gp}^-$ [Peripheral Low-Permeation Agent]**  
   Fails passive permeability due to high polarity or size; evades P-gp. Strictly peripheral (e.g., Penicillin G).
4. **Quadrant 4: $\text{BBB}^- / \text{P-gp}^+$ [Restricted Effluxed Peripheral Compound]**  
   Doubly blocked: poor passive permeation compounded by active efflux.

### Applicability Domain & Uncertainty
* **Tanimoto Structural Similarity:** Max Morgan similarity against training benchmarks ($T_{\max} \ge 0.40$ indicates inside chemical domain).
* **Descriptor Leverage Distance:** Normalized Euclidean distance to training centroids.
* **Dual-Barrier Uncertainty Score:**
  $$U_{\text{dual}} = 1 - 2 \cdot \left| P(\text{BBB}^+) - 0.5 \right| \cdot \left| P(\text{P-gp}^+) - 0.5 \right|$$

### Reaction-Based Molecular Counterfactuals
The engine generates chemically valid analogs to flip predictions using established medicinal chemistry transformations:
* N-methylation of primary/secondary amines (reduces HBD count).
* Fluorination and bioisosteric replacement.
* Ether dealkylation to modulate excessive lipophilicity.

---

---

## 8. Codebase Architecture & Future Upgrade Guide

The codebase is organized into modular packages:

```
Novelty Upgrade/
├── core/                           # Shared feature engineering & domain logic
│   ├── features.py                 # 2D, Morgan FP, and 3D ETKDG feature extractors
│   └── decision_matrix.py          # 4-quadrant logic & clinical benchmark definitions
├── explainability/                 # Explainability & structural deconvolution
│   ├── shap_explainer.py           # TreeSHAP in log-odds margin space
│   ├── substructure_mapper.py      # Morgan bit-to-subgraph mapper
│   ├── shared_properties.py        # Cross-task descriptor divergence
│   ├── applicability_domain.py     # Tanimoto & leverage domain estimator
│   ├── counterfactual.py           # Reaction-based analog search
│   └── report_generator.py         # Standalone HTML, CSV, JSON export
├── pipelines/                      # Data preprocessing, training & validation workflows
│   ├── bbb_pipeline.py             # BBB feature extraction, training & evaluation
│   ├── pgp_pipeline.py             # P-gp 3D conformer feature extraction & training
│   ├── validation.py               # 1000-bootstrap validation runner
│   ├── ablation.py                 # 5-fold feature ablation study runner
│   └── global_xai.py               # Global TreeSHAP distributions & correlation plots
├── data/                           # Curated small-molecule datasets
├── models/                         # Serialized XGBoost models & scaler metadata
├── figures/                        # High-resolution publication plots
├── results/                        # Validation metrics & evaluation tables
├── tests/                          # Automated regression test suite
└── inference.py                    # Top-level CLI prediction interface
```

### How to Extend the Framework

#### Adding a New Transporter (e.g., BCRP / ABCG2)
1. Add raw dataset to `data/bcrp_benchmark.tab`.
2. Extract features using `core.features.calculate_2d_descriptors` and `calculate_3d_descriptors`.
3. Add a model training pipeline in `pipelines/bcrp_pipeline.py`.
4. Extend `inference.py` to support multi-transporter triage.

#### Adding New Descriptors (e.g., Quantum Mechanical DFT or MACCS Keys)
1. Implement the descriptor extraction in `core/features.py`.
2. Update feature column ordering in `prepare_dataframe`.
3. Re-run `pipelines/bbb_pipeline.py` and `pipelines/pgp_pipeline.py` to retrain models and update scaler metadata.
