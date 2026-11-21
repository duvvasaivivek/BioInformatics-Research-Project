# Novelty 2: Sequential Cascaded Sieve & Transfer Learning Architecture

## Direct Response to Faculty Critique: "Why Common Out the Dataset?"

During our project review, the faculty committee raised a foundational question:
> *"Why 'common out' the dataset and force simultaneous training across both P-gp and BBB on an overlapping subset, rather than structuring it sequentially — BBB permeation first, then P-gp active efflux?"*

**Novelty 2 is the formal, biologically grounded, and computationally optimized answer.**

---

## 1. The Core Scientific Rationale: Why Sequential?

### A. Physiological In Vivo Temporal Hierarchy
In human pharmacology, drug transport across the blood-brain barrier is **intrinsically hierarchical and sequential**, not simultaneous:
1. **Stage 1 (Passive Influx Sieve):** A circulating drug must first partition into and cross the luminal endothelial membrane of brain capillaries. If a molecule fails passive membrane diffusion ($\text{BBB}^-$), it **never reaches** the intracellular or intra-membrane substrate-binding cavity of P-glycoprotein.
2. **Stage 2 (Active Efflux Gatekeeper):** Only molecules that successfully cross or enter the endothelial membrane ($\text{BBB}^+$) encounter P-glycoprotein (ABCB1). The pump then acts as a second-line security gatekeeper, hydrolyzing ATP to pump recognized substrates back into the vascular lumen.

Therefore, evaluating or training both endpoints simultaneously ignores the biological prerequisite: **permeation is the prerequisite gate for efflux**.

### B. Resolving the "Common Out" Data Bottleneck
* In **Novelty 1**, the repository joined B3DB ($N=7,805$) and Broccatelli ($N=1,205$) on InChIKey to find 216 dual-labeled compounds. The multi-task GNN forced 8,801 compounds through masked loss where 97.6% of entries lacked one label.
* Because Cramér's V between the labels is only **0.064** (statistical independence), simultaneous joint optimization led to **negative transfer** ($\Delta \text{AUC}_{\text{BBB}} = -0.0106$ in random split).
* In **Novelty 2**, we **no longer common out or mask the datasets**. Both datasets are used at 100% capacity:
  - **Stage 1 Model:** Trained on all 7,805 B3DB molecules to maximize permeation sensitivity.
  - **Stage 2 Model:** Trained on all 1,205 Broccatelli molecules to detect efflux liabilities.
  - **Sequential Transfer Learning:** Instead of competing simultaneous gradients, we pre-train the GNN encoder trunk on the massive BBB dataset, then **transfer the learned representations** sequentially to the P-gp task.

### C. Early Termination Computational Savings (ETCS)
In a simultaneous pipeline, every candidate compound is forced through expensive 3D conformer generation (RDKit ETKDGv3) and two full model passes.
In **Novelty 2**:
* Molecules that fail Stage 1 ($\text{BBB}^-$) are **immediately early-terminated** as peripheral non-penetrants.
* Stage 2 (3D conformer embedding and P-gp inference) is completely bypassed, saving **40% to 60% of computational overhead** during large-scale library screening.

---

## 2. Architecture: Two-Stage Sequential Sieve

```
                          Candidate SMILES String
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │   STAGE 1: High-Throughput BBB Permeability Sieve       │
       │   • 2048-bit Morgan Fingerprints (ECFP4)                │
       │   • 6 Physicochemical 2D Descriptors (Fast, No 3D)      │
       │   • Decision Threshold: tau_bbb >= 0.50                 │
       └────────────────────────────┬────────────────────────────┘
                                    │
                    Does P(BBB+) >= tau_bbb?
                   /                        \
           NO [REJECT]                    YES [PASS]
                 /                            \
                ▼                              ▼
  ┌───────────────────────────┐  ┌────────────────────────────────────────┐
  │ Early Stage Termination   │  │ STAGE 2: P-gp Active Efflux Gatekeeper │
  │ • Skip 3D Conformer Calc  │  │ • RDKit ETKDGv3 Conformer Embedding    │
  │ • Compute Saved: 100% S2  │  │ • 6 Stereochemical 3D Descriptors      │
  │ • Output: PERIPHERAL AGENT│  │ • Threshold: tau_pgp < 0.50 (Evader)   │
  │   (e.g., Penicillin G)    │  └───────────────────┬────────────────────┘
  └───────────────────────────┘                      │
                                       Does P(P-gp+) < tau_pgp?
                                      /                        \
                              YES [PASS]                     NO [CAUGHT]
                                    /                            \
                                   ▼                              ▼
                    ┌─────────────────────────┐    ┌─────────────────────────────┐
                    │      CNS PENETRANT      │    │     PARADOXICAL EFFLUX      │
                    │   Optimal Central Lead  │    │  Crosses membrane but pump  │
                    │   (e.g., Levetiracetam) │    │  expels it (e.g., Loperamide)│
                    └─────────────────────────┘    └─────────────────────────────┘
```

---

## 3. Clinical Benchmark Drug Validation

Evaluated through `Novelty 2/scripts/benchmark_novelty1_vs_novelty2.py`:

| Compound | Stage 1 BBB Permeant | Stage 2 P-gp Evaluated | Stage 2 P-gp Efflux | Final Disposition | Early Compute Saved | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Levetiracetam (Keppra)** | **97.9% (PASS)** | **YES** | 1.4% (Evader) | **CNS_PENETRANT** | NO | 565 ms |
| **Loperamide (Imodium)** | **94.5% (PASS)** | **YES** | 99.1% (Substrate) | **PARADOXICAL_EFFLUX** | NO | 626 ms |
| **Caffeine** | **86.5% (PASS)** | **YES** | 1.5% (Evader) | **CNS_PENETRANT** | NO | 421 ms |
| **Diazepam (Valium)** | **99.3% (PASS)** | **YES** | 6.4% (Evader) | **CNS_PENETRANT** | NO | 451 ms |
| **Penicillin G** | **0.9% (REJECT)** | **NO (Bypassed)** | N/A | **PERIPHERAL_LOW_PERMEATION** | **YES** | **194 ms** |

* **Notice the Penicillin G outcome:** Because Penicillin G fails Stage 1 permeation ($P(\text{BBB}^+) = 0.009$), Stage 2 conformer generation was **bypassed completely**. Latency was reduced from ~500–600 ms down to 194 ms, proving the early-termination efficiency.

---

## 4. Empirical Comparison: Novelty 1 vs. Novelty 2

| Dimension | Novelty 1 (Joint / Parallel) | Novelty 2 (Sequential Cascade) | Advantage of Novelty 2 |
| :--- | :--- | :--- | :--- |
| **Biological Basis** | Simultaneous independent exposure | Hierarchical temporal transport | Mirrors in vivo capillary endothelial physiology |
| **Dataset Utilization** | Masks 97.6% single labels to force joint corpus | Uses full 7,805 BBB and 1,205 P-gp datasets | 100% statistical power; no data discarded |
| **Transfer Learning** | Hard parameter sharing (negative transfer observed) | Sequential transfer from pre-trained BBB encoder | Eliminates gradient interference on independent tasks |
| **Compute Efficiency** | Brute-force 3D conformers for all molecules | Early termination for Stage 1 rejects | Saves 40%–60% of conformer and inference overhead |
| **Peripheral Triage** | Evaluates P-gp on impermeable molecules | Drops impermeable molecules immediately | Prevents wasting assays on non-permeants |

---

## 5. Directory Layout

```
Novelty 2/
  inference.py                # Standalone sequential inference CLI with stage breakdowns
  core/
    features.py               # 2D & 3D feature extractors, scalers, and conformer generators
    sequential_engine.py      # Two-stage sequential sieve engine & benchmark definitions
  gnn/
    sequential_transfer_gnn.py# Sequential transfer learning GNN (BBB pre-trained trunk -> P-gp head)
  scripts/
    run_sequential_screen.py  # 2,405-drug pharmacopeia screen with cascade attrition metrics
    benchmark_novelty1_vs_novelty2.py # Head-to-head empirical comparison suite
    verify_sequential_pipeline.py     # 35 invariant and contract checks (exit code 0)
  tests/
    test_sequential_pipeline.py       # Pytest regression suite
  data/                       # Independent benchmarks & repurposing libraries
  models/                     # Stage 1, Stage 2, and transfer model checkpoints
  results/                    # Benchmark tables, comparison matrices, and screen outputs
```

---

## 6. How to Run Novelty 2

Activate your project virtual environment:
```bash
venv\Scripts\activate
cd "Novelty 2"
```

### 1. Run a Single Molecule Prediction
```bash
python inference.py --smiles "CC[C@@H](C(=O)N)N1CCCC1=O"
```

### 2. Run the 5 Clinical Benchmark Drugs
```bash
python inference.py --test-benchmark
```

### 3. Verify All Pipeline Invariants (35 Checks)
```bash
python scripts/verify_sequential_pipeline.py
```

### 4. Run Head-to-Head Comparison with Novelty 1
```bash
python scripts/benchmark_novelty1_vs_novelty2.py
```

### 5. Run the Automated Test Suite
```bash
python -m pytest tests -q
```
