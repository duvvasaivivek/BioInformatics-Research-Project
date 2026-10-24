# MICRO-BLOCK ARCHITECTURAL FLOW SPECIFICATION
## Dual-Barrier CNS Drug Permeation & Efflux Engine — Every Operation, Every Tensor, Every Objective

> **Purpose of this document.** This is not a top-level model summary. Every single computational
> block in the repository is enumerated as a numbered node with its exact input, exact output,
> exact shape/dtype, and the loss or objective function attached to it (or an explicit
> "none — deterministic" when a block has no learnable objective). It is written so you can
> hand-draw the full architecture from it without reopening the source code.
>
> **Generated:** 2026-10-24 · **Audited commit:** `a53789f` (Restructured Codebase) · **Branch:** `main`
> **Companion docs:** `ARCHITECTURE_DEEP_DIVE.md` (narrative/biology), `DATASET_AND_FILE_PROVENANCE_CATALOG.md` (data lineage)

---

## HOW TO READ THIS DOCUMENT

### Block ID scheme

Every block has a stable ID you can write inside the box when you draw it:

| Prefix | Subsystem |
|---|---|
| `A#` | **Paradigm A** — Initial Model, end-to-end Graph Neural Network |
| `B#` | **Paradigm B** — B3DB Dual-Barrier XGBoost engine (feature core, training, inference) |
| `C#` | **Explainability & audit layer** (TreeSHAP, substructure, applicability domain, counterfactual, report) |
| `D#` | **Scientific validation layer** (bootstrap CI, scaffold split, ablation, screening, docking) |

### Block card legend

Each table row is one drawable box. Columns mean:

- **IN** — what enters the box, with shape and dtype.
- **OUT** — what leaves the box, with shape and dtype.
- **LOSS / OBJECTIVE** — the optimisation target *attached to that specific block*. Most blocks are
  deterministic transforms and carry no loss; those are marked `none — deterministic`. Only four
  blocks in the entire repository carry a true trainable loss (A7.4, B4.3, B4.5, B5.4), plus two
  non-differentiable search objectives (B4.4, C5.3) and one empirical physics score (D4.5).
- **SHAPE NOTATION** — `N` = number of molecules, `V` = atoms in one molecule, `E` = bonds,
  `B` = batch size, `H` = hidden width.

### Symbol key for the drawing

All diagrams in this document are plain-text flow-and-block drawings. Every diagram uses exactly
this notation, and the right-hand column tells you what shape to draw when you redraw it by hand.

```text
  NOTATION IN THIS DOC          MEANING                              DRAW IT AS

  +------------------+
  |  B4.2a  name     |          deterministic transform, no loss     plain rectangle
  +------------------+

  +==================+
  |  (o) B4.4b LOSS  |          learnable block / loss boundary      double-ruled rectangle
  +==================+

  (o)  B4.4b                    loss or objective NODE               hexagon
  <>   A7.2                     branch / decision                    diamond
  (raw) filename.json           file on disk, intermediate           parallelogram
  (db)  filename.csv            persisted dataset or model artifact  cylinder
  /!\                           audit flag -- something is wrong     red annotation

  ARROWS
     |      v                   tensor / array dataflow              solid arrow
     :    .....>                gradient, metric, or control signal  dashed arrow
     //====  ====\\             loop boundary                        bracket / back-arrow
     \\====  ====//

  SHAPE ANNOTATIONS
  [V x 9]                       tensor shape, V = atoms             write ON the arrow
  [1 x 2054]                    DataFrame shape                     write ON the arrow
  72 par                        learnable parameter count           write INSIDE the box
```

Boxes carry only an ID and a short name. The full input, output, shape, dtype and loss for every
block is in the table immediately beneath each diagram — diagram and table are meant to be read
side by side.

In the prose and tables outside the diagrams, the same artifacts appear as shape glyphs instead of
ASCII tags: ▱ is `(raw)` a file on disk, ⬓ is `(db)` a persisted dataset or model, ⬡ is `(o)` a
loss node, ◇ is `<>` a branch. They mean the same thing; the diagrams stay plain ASCII so they
survive copy-paste into any terminal, editor or whiteboard photo.

---

## SECTION 0 — REPOSITORY MAP

### 0.1 Physical layout

```
BioInformatics Project/
+-- Initial Model/                        <-- PARADIGM A (GNN, superseded but retained)
|   +-- Dataset/
|   |   +-- raw_adenot_li.xls             (raw) raw 2-sheet Excel source
|   |   +-- cleaned_bbb_dataset.csv       (db) 1,890 curated molecules
|   |   +-- graph_summary.csv             (db) per-graph node/edge census
|   +-- clean_raw_data.py                 A1-A3  ingest + curate + render
|   +-- graph_generation.py               A4-A5  featurise + serialise graphs
|   +-- gnn_dataset.py                    A6     PyG Dataset adapter
|   +-- check_graph.py                    A6.4   single-graph inspector
|   +-- train_gnn.py                      A7-A9  CV harness + model + eval
|   +-- predict.py                        A10    single-SMILES inference CLI
|
+-- B3DB Based Model/                     <-- PARADIGM B (deployed dual-barrier engine)
|   +-- core/
|   |   +-- features.py                   B3     shared feature kernel
|   |   +-- decision_matrix.py            B8     four-quadrant logic + benchmarks
|   +-- BBB/
|   |   +-- bbb_data_preprocessing.py     B1     B3DB > feature matrix
|   |   +-- bbb_model_training.py         B4     Optuna + XGBoost (deployed)
|   +-- Pgp/
|   |   +-- pgp_data_preprocessing.py     B2     Broccatelli > 3D feature matrix
|   |   +-- pgp_model_training.py         B5     XGBoost (deployed)
|   +-- explainability/
|   |   +-- shap_explainer.py             C1     TreeSHAP margin decomposition
|   |   +-- substructure_mapper.py        C2     Morgan bit > fragment SMILES
|   |   +-- applicability_domain.py       C3     Tanimoto + leverage gating
|   |   +-- shared_properties.py          C4     cross-task comparison
|   |   +-- counterfactual.py             C5     reaction-SMARTS analog search
|   |   +-- report_generator.py           C6     CSV / JSON / HTML export
|   +-- scripts/
|   |   +-- run_validation.py             D1     bootstrap CI + scaffold split
|   |   +-- run_ablation_study.py         D2     feature-subset ablation
|   |   +-- run_explainability_analysis.py D3    global SHAP + figures
|   |   +-- prepare_repurposing_library.py D4.0  Broad Hub download/merge
|   |   +-- run_repurposing_screen.py     D4     2,405-drug virtual screen
|   |   +-- run_cryoem_docking_validation.py D5  AutoDock Vina into 6QEX
|   +-- tests/test_explainability.py      D6     4-quadrant regression suite
|   +-- data/ models/ figures/ results/   (db) artifacts
|   +-- inference.py                      B9     orchestrating CLI entry point
+-- requirements.txt
```

### 0.2 Exact dataset cardinalities (measured from the files on disk, not from prose)

| Artifact | Rows | Columns | Notes |
|---|---|---|---|
| `data/B3DB_classification.tsv` | 7,809 | many | raw B3DB, tab-separated |
| `data/processed_b3db_data_advanced.csv` | 7,805 | 2,056 | 4 molecules lost to RDKit parse failure |
| `data/pgp_broccatelli.tab` | 1,219 | 2 | raw Broccatelli/TDC |
| `data/processed_pgp_data.csv` | 1,205 | 2,062 | 14 molecules lost to 3D embedding failure |
| `data/repurposing/fda_approved_repurposing_compounds.csv` | 2,405 | 7 | Launched-phase only |
| `Initial Model/Dataset/cleaned_bbb_dataset.csv` | 1,890 | 9 | Adenot + Li merged |

> **Column arithmetic.** `processed_b3db` = `SMILES` + `Label` + `FP_0…FP_2047` + 6 descriptors
> = 2 + 2048 + 6 = **2,056 columns**, of which **2,054 are model features**.
> `processed_pgp` = 2 + 2048 + 12 = **2,062 columns**, of which **2,060 are model features**.
> These two numbers are confirmed inside the saved boosters: `num_feature = 2054` and `2060`.

---

## SECTION 1 — MASTER FLOW (the one-page skeleton)

Draw this first, then expand each shaded cluster using Sections 2–6.

```text
                                   +--------------------------------+
                                   |   INPUT * SMILES string        |
                                   |   one molecule, or a dataset   |
                                   +---------------+----------------+
            +----------------------------------------------------------------+
            |            NO TENSOR EVER CROSSES THIS DIVIDE                  |
            v                                                                v
+==========================================+    +==========================================+
|  PARADIGM A * Initial Model GNN          |    |  PARADIGM B * Dual-Barrier Engine        |
|  learned representation                  |    |  engineered representation               |
+==========================================+    +==========================================+
            |                                                                |
+-----------v------------------------------+    +-----------v------------------------------+
| A1-A3   Ingest + curate                  |    | B1 / B2   Preprocess to feature matrix   |
|   1,890 curated                          |    |   BBB  7,805 x 2056                      |
|   1,621 actually trained (Li filtered)   |    |   Pgp   1,205 x 2062                     |
+-----------+------------------------------+    |   RAW descriptors on disk = leakage guard|
            |                                   +-----------+------------------------------+
+-----------v------------------------------+                |
| A4-A5   Graph featurisation              |    +-----------v------------------------------+
|   x          [V  x 9]                    |    | B3   Shared feature kernel               |
|   edge_index [2  x 2E]                   |    |   Morgan 2048 | 2D x6 | 3D x6 (ETKDGv3)  |
|   edge_attr  [2E x 7]                    |    +----+--------------------------+----------+
+-----------+------------------------------+         | 2,054                    | 2,060
            |                                        v                          v
+-----------v------------------------------+    +---------------------+  +---------------------+
| A7-A8   3x GINEConv + MLP readout        |    | B4  XGBoost BBB     |  | B5  XGBoost P-gp    |
|   345,514 parameters                     |    |   Optuna 50 trials  |  |   untuned baseline  |
|   receptive field = 3 bonds              |    |   494 trees, ES 50  |  |   300 trees, no ES  |
+-----+-------------------------+----------+    +----------+----------+  +----------+----------+
      |                         ^                          | P(BBB+)                | P(P-gp+)
      v                         : gradient                  +-----------+-----------+
+--------------------------+    :                                       |
| (o) A7.12   LOSS         |----+               +-----------------------v------------------+
|   weighted softmax       |                    | (o) B4.4b / B4.5a / B5.5   LOSS          |
|   cross-entropy          |                    |   weighted binary logistic               |
+------------+-------------+                    |   + Omega(gamma, lambda, alpha) penalty  |
             |                                  +-----------------------+------------------+
+------------v-----------------------------+                            |
| A10   Inference REPL                     |    +-----------------------v------------------+
|   P(BBB+), P(BBB-)                       |    | B8   Four-quadrant decision matrix       |
|   [runs on fold-1 weights]               |    |   S_CNS = P(BBB+) x (1 - P(P-gp+))       |
+------------------------------------------+    +-----------------------+------------------+
                                                                        |
   PARADIGM A HAS NO C-LAYER                           +----------------+----------------+
   PARADIGM A HAS NO D-LAYER                           v                                 v
   (draw that asymmetry - it is itself      +-----------------------+      +-----------------------+
    a finding)                              | C-LAYER   Explain     |      | D-LAYER   Validate    |
                                            |   C1  TreeSHAP        |      |   D1    Bootstrap CI  |
                                            |   C2  bit->fragment   |      |   D1.4  Scaffold split|
                                            |   C3  Applicability   |      |   D2    Ablation      |
                                            |   C4  Cross-task      |      |   D3    Global SHAP   |
                                            |   C5  Counterfactual  |      |   D4    Screen 2,405  |
                                            |   C6  CSV/JSON/HTML   |      |   D5    Vina -> 6QEX  |
                                            +-----------------------+      +----------+------------+
                                                                                      :
                             D5 feeds back to B5  <----------------------------------- +
                             "orthogonal structural validation" (3 caveats, see 6.11)
```

### 1.1 The single most important structural fact

The two paradigms **never exchange tensors**. They share only the input SMILES string and the
biological question. Paradigm A is a learned-representation model (the molecule's representation is
discovered by message passing). Paradigm B is a fixed-representation model (the representation is
hand-engineered, then a tree ensemble is fitted). When you draw the master diagram, draw them as
two parallel vertical columns with **no horizontal arrows between them**.

---

## SECTION 2 — PARADIGM A: INITIAL MODEL GRAPH NEURAL NETWORK

### 2.1 Stage A1–A3 · Data ingestion and curation

Source: `Initial Model/clean_raw_data.py`

```text
                        (raw) raw_adenot_li.xls   -- two sheets
                                     |
                     +---------------+---------------+
                     v                               v
          +--------------------+          +--------------------+
          | A1.1  read_excel   |          | A1.2  read_excel   |
          |   Table S1-Adenot  |          |   Table S2-Li      |
          |   header = 2       |          |   header = 2       |
          +---------+----------+          +---------+----------+
                    v                               v
          +--------------------+          +--------------------+
          | A2.1  project 4    |          | A2.2  project 7    |
          |   cols + rename    |          |   cols + rename    |
          |   Source="Adenot"  |          |   Source="Li"      |
          +---------+----------+          +---------+----------+
                    +---------------+---------------+
                                    v
                      +-------------------------------+
                      | A2.3  Schema harmonise        |
                      |   pad missing cols with None  |
                      |   reorder to 9-col schema     |
                      +---------------+---------------+
                                      v
                      +-------------------------------+
                      | A2.4  Vertical concat         |
                      |   ignore_index = True         |
                      +---------------+---------------+
                                      v
                      +-------------------------------+
                      | A2.5  dropna(SMILES, BBB)     |
                      +---------------+---------------+
                                      v
                      +-------------------------------+
                      | A2.6  drop_duplicates(SMILES) |
                      |   keeps FIRST occurrence, so  |
                      |   Adenot wins every collision |
                      +---------------+---------------+
                                      v
                            <> A2.7  label coercion
                            |
                            +-- "0/1" ---------> 0      AMBIGUOUS -> NEGATIVE
                            |                           (conservative curation)
                            +-- else ----------> int(float(v))
                                      |
                                      v
                      +-------------------------------+
                      | A2.8  MolFromSmiles           |
                      +---------------+---------------+
                                      v
                      +-------------------------------+
                      | A2.9  dropna(Molecule)        |
                      |   OUT: 1,890 valid molecules  |
                      +----+---------------------+----+
                           |                     |
                           v                     v
          +-------------------------+   +-------------------------------------+
          | A2.10  to_csv           |   | A3   IMAGE BRANCH  -- DEAD END      |
          |  (db) cleaned_bbb_      |   |   A3.1  MolToImage 700x500          |
          |       dataset.csv       |   |   A3.2  canvas 1000x1000 + 15 lines |
          +------------+------------+   |   A3.3  save molecule_NNNN.png      |
                       |                |   (raw) 1,890 PNGs -- feeds NOTHING |
                       v                +-------------------------------------+
            on to A4 featurisation
```

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **A1.1** | `pd.read_excel(sheet="Table S1-Adenot", header=2)` | `.xls` binary | DataFrame, raw Adenot table | none — deterministic |
| **A1.2** | `pd.read_excel(sheet="Table S2-Li", header=2)` | `.xls` binary | DataFrame, raw Li table | none — deterministic |
| **A2.1** | Project `SMILES, SMILES-B, Name, BBB-crossing`; rename to `SMILES, Alternative_SMILES, Name, BBB`; tag `Source="Adenot"` | Adenot DF | DF `[n_A × 5]` | none — deterministic |
| **A2.2** | Project `No, No-Li, Name, SMILES-Li, SMILES, Class, Ref.`; rename; tag `Source="Li"` | Li DF | DF `[n_L × 8]` | none — deterministic |
| **A2.3** | Insert `None` for columns absent in each source, reorder to canonical 9-column schema | 2 DFs | 2 schema-identical DFs | none — deterministic |
| **A2.4** | `pd.concat(ignore_index=True)` | 2 DFs | DF `[n_A+n_L × 9]` | none — deterministic |
| **A2.5** | `dropna(subset=["SMILES","BBB"])` | DF | DF, label-complete rows | none — deterministic |
| **A2.6** | `drop_duplicates(subset=["SMILES"])` — **keeps first occurrence, so Adenot wins every collision** | DF | DF, unique canonical SMILES | none — deterministic |
| **A2.7** | `convert_label`: `"0/1" → 0`, else `int(float(v))` | str/float | `int ∈ {0,1}` | none — deterministic |
| **A2.8** | `Chem.MolFromSmiles` per row | SMILES str | RDKit `Mol` or `None` | none — deterministic |
| **A2.9** | `dropna(subset=["Molecule"])` | DF | **1,890 valid molecules** | none — deterministic |
| **A2.10** | `to_csv` with `Molecule` column dropped | DF `[1890 × 9]` | ⬓ `cleaned_bbb_dataset.csv` | none — I/O |
| **A3.1** | `Draw.MolToImage(mol, size=(700,500))` | `Mol` | RGB PIL image `700×500×3` | none — deterministic |
| **A3.2** | White `1000×1000` canvas, paste structure at `(150,70)`, draw 15 metadata lines from `y=590` with 85-char wrap | image + row dict | annotated RGB `1000×1000×3` | none — deterministic |
| **A3.3** | `canvas.save("images/molecule_NNNN.png")` | image | ▱ 1,890 PNG files | none — I/O |

> **Drawing note for A2.7.** This block is biologically load-bearing and easy to miss. Ambiguous
> `"0/1"` literature labels are coerced to the **negative** class (`0`), not dropped. Draw it as a
> diamond with two exits: `"0/1" → 0` and `else → int(float(v))`. It makes the curation
> conservative: an uncertain permeant is treated as a non-permeant.

> **Drawing note for A3.** The entire A3 branch is a **side output**. It produces human-readable
> PNGs and feeds nothing downstream. Draw it hanging off A2.9 as a dead-end stub so the reader
> immediately sees it is not in the learning path.

---

### 2.2 Stage A4–A5 · Graph featurisation and serialisation

Source: `Initial Model/graph_generation.py`

```text
     (db) cleaned_bbb_dataset.csv  ->  one row: SMILES, BBB, Name, Source
                      |
                      v
          +---------------------------+
          | A4.1  MolFromSmiles       |
          |   continue if None        |
          +------+-------------+------+
                 |             |
                 v             v
      +------------------+  +------------------+
      | A4.2  atom       |  | A4.3  bond       |
      |   features       |  |   features       |
      |   9 per atom     |  |   7 per bond     |
      +--------+---------+  +--------+---------+
               |                     v
               |          +---------------------------+
               |          | A4.4  edge duplication    |
               |          |   append [s,d] AND [d,s]  |
               |          |   same 7-vector for both  |
               |          |   E  ->  2E directed      |
               |          +---------+-----------------+
               +-----------+--------+
                           v
          +-------------------------------------------+
          | A4.5  Tensor assembly                     |
          |   x          [V  x 9 ]   float32          |
          |   edge_index [2  x 2E]   int64  (.t())    |
          |   edge_attr  [2E x 7 ]   float32          |
          |   y          scalar      int64            |
          +--------+-------------------------+--------+
                   |                         |
                   v                         v
      +-------------------------+  +-------------------------+
      | A5.1  torch.save dict   |  | A5.2  census row        |
      |  (raw) graph_NNNN.pt    |  |   ID, SMILES, BBB,      |
      |   + smiles/name/source  |  |   Nodes, Edges          |
      +-------------------------+  +------------+------------+
                                                v
                                   +-------------------------+
                                   | A5.3  to_csv            |
                                   |  (db) graph_summary.csv |
                                   +-------------------------+
```

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **A4.1** | `Chem.MolFromSmiles`, `continue` on `None` | SMILES str | `Mol` | none — deterministic |
| **A4.2** | `get_atom_features` → 9 scalars per atom | `Atom` | `list[9]` | none — deterministic |
| **A4.3** | `get_bond_features` → 7 scalars per bond | `Bond` | `list[7]` | none — deterministic |
| **A4.4** | Append `[src,dst]` **and** `[dst,src]`, duplicating the same 7-vector for both directions | `E` bonds | `2E` directed edges | none — deterministic |
| **A4.5** | `x = tensor(float32)` `[V,9]`; `edge_index = tensor(int64).t().contiguous()` `[2,2E]`; `edge_attr = tensor(float32)` `[2E,7]` | lists | 3 tensors | none — deterministic |
| **A5.1** | `torch.save({x, edge_index, edge_attr, y, smiles, name, source})` | tensors + metadata | ▱ `graph_NNNN.pt` | none — I/O |
| **A5.2–A5.3** | Census of `Nodes = x.shape[0]`, `Edges = edge_index.shape[1]` | per-graph | ⬓ `graph_summary.csv` | none — I/O |

#### A4.2 · Node feature vector, exact index order (`d_v = 9`)

Draw this as a labelled 9-cell column vector. Index order is **load-bearest**: `predict.py` must
reproduce it exactly or the first `Linear(9, 256)` silently consumes permuted chemistry.

| Idx | RDKit call | Semantic | Range | Encoding |
|---|---|---|---|---|
| 0 | `GetAtomicNum()` | element identity | 1–53 in this corpus | **raw integer, not one-hot** |
| 1 | `GetDegree()` | heavy-atom neighbours | 0–4 | raw integer |
| 2 | `GetFormalCharge()` | formal charge | −1…+1 | raw signed integer |
| 3 | `GetTotalNumHs()` | attached hydrogens | 0–4 | raw integer |
| 4 | `GetIsAromatic()` | aromaticity | {0,1} | boolean → int |
| 5 | `IsInRing()` | ring membership | {0,1} | boolean → int |
| 6 | `GetExplicitValence()` | explicit valence | 0–6 | raw integer |
| 7 | `GetImplicitValence()` | implicit valence | 0–4 | raw integer |
| 8 | `int(GetHybridization())` | hybridisation enum | 0–7 | **ordinal cast of an enum** |

> **Audit flag (draw as a red annotation).** Indices 0 and 8 are categorical quantities fed as raw
> ordinals, and **no feature normalisation is applied anywhere** in Paradigm A. Atomic number 53
> (iodine) enters the first linear layer ~53× larger than a boolean flag. The first `BatchNorm1d`
> sits *after* `conv1`, so it normalises the already-mixed message-passing output, not the raw
> input. This is the single largest representational weakness of Paradigm A and a direct reason
> Paradigm B's standardised descriptors behave better.

#### A4.3 · Edge feature vector, exact index order (`d_e = 7`)

| Idx | Expression | Semantic | Encoding |
|---|---|---|---|
| 0 | `bond_type == SINGLE` | single bond | one-hot slot |
| 1 | `bond_type == DOUBLE` | double bond | one-hot slot |
| 2 | `bond_type == TRIPLE` | triple bond | one-hot slot |
| 3 | `bond_type == AROMATIC` | aromatic bond | one-hot slot |
| 4 | `GetIsAromatic()` | aromatic flag | boolean |
| 5 | `GetIsConjugated()` | conjugation | boolean |
| 6 | `IsInRing()` | ring bond | boolean |

Indices 0–3 form a proper mutually-exclusive one-hot block; index 4 is **redundant with index 3**
(aromatic bond type implies aromatic flag). Harmless for a tree-free model but worth annotating.

---

### 2.3 Stage A6 · Dataset adapter

Source: `Initial Model/gnn_dataset.py`, `Initial Model/check_graph.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **A6.1** | `BBBGraphDataset.__init__` — `sorted(listdir)` filtered to `*.pt` | folder path | file index list | none — deterministic |
| **A6.2** | `len()` → `len(self.files)` | — | int | none — deterministic |
| **A6.3** | `get(idx)` — `torch.load(weights_only=False)`, wrap into PyG `Data`, attach `smiles`/`name`/`source` | `.pt` path | `Data` object | none — deterministic |
| **A6.4** | `check_graph.py` — prints `x`, `edge_index`, `edge_attr` and their shapes for `graph_0001.pt` | `.pt` | stdout | none — diagnostic |

> **Important:** `gnn_dataset.py` is a **standalone demonstration/inspection script**, not a
> dependency of training. `train_gnn.py` does **not** import `BBBGraphDataset`; it re-implements
> loading inline (block A7.1) so it can apply the source filter A7.2. Draw A6 as a parallel
> diagnostic branch, not as an inline stage of the training pipeline.

---

### 2.4 Stage A7 · Training harness

Source: `Initial Model/train_gnn.py`

```text
   (raw) graphs/*.pt   -- 1,890 files on disk
            |
            v
+-------------------------------------------+
| A7.1  Load all, cast dtypes               |
|   try/except -> invalid_graphs++          |
+--------------------+----------------------+
                     v
            <> A7.2  SOURCE FILTER
            |
            +-- source == "Li" --> DISCARD    269 graphs (14.2%)
            |
            +-- else ------------> KEEP     1,621 Adenot
                     |
                     v
+-------------------------------------------+
| A7.3  labels[] int64 + class histogram    |
+--------------------+----------------------+
                     v
+-------------------------------------------+
| A7.4  StratifiedKFold                     |
|   n_splits=10, shuffle=True, seed=42      |
+--------------------+----------------------+
                     |
      //==================== PER FOLD, x10 ====================\\
                     v
+-------------------------------------------+
| A7.5  Inner split 90/10 stratified        |
|   -> train_idx , val_idx                  |
+--------------------+----------------------+
                     v
+-------------------------------------------+
| A7.6  Class weights  w_c = N / (2 * n_c)  |
+--------------------+----------------------+
                     v
+-------------------------------------------+
| A7.7  DataLoaders x3   batch_size = 32    |
|   shuffle=True on TRAIN ONLY              |
+--------------------+----------------------+
                     v
+-------------------------------------------+
| A7.8   FRESH BBB_GNN  (no carry-over)     |
| A7.9   AdamW  lr=5e-4  weight_decay=5e-4  |
| A7.10  ReduceLROnPlateau  mode="max"      |
|          factor=0.5 patience=8 min=1e-6   |
+--------------------+----------------------+
                     v
       //---------- A7.11 epoch loop, max 200 ----------\\
                     v                                  ||
      +----------------------------+                    ||
      | (o) A7.12   LOSS           |                    ||
      |   weighted softmax         |                    ||
      |   cross-entropy            |                    ||
      +-------------+--------------+                    ||
                    v                                   ||
      +----------------------------+                    ||
      | A7.13  loss.backward()     |                    ||
      |   clip_grad_norm_(2.0)     |                    ||
      |   optimizer.step()         |                    ||
      +-------------+--------------+                    ||
                    v                                   ||
      +----------------------------+                    ||
      | A7.14  evaluate(val)       |                    ||
      |   acc prec rec F1 ROC-AUC  |                    ||
      +-------------+--------------+                    ||
                    |                                   ||
                    +...... val F1 ......> A7.10 (LR)   ||
                    |                                   ||
                    v                                   ||
      +----------------------------+                    ||
      | A7.15  checkpoint IF       |                    ||
      |   val F1 improved          |                    ||
      |   (raw) ..._fold_K.pt      |                    ||
      |   SELECTION METRIC = F1    |                    ||
      +-------------+--------------+                    ||
                    v                                   ||
            <> A7.16  patience >= 20 ?                  ||
            |                                           ||
            +-- no  -----------------------------------=//
            |
            +-- yes --> break
                    |
                    v
+-------------------------------------------+
| A7.17  reload best, score fold TEST       |
+--------------------+----------------------+
                     |
      \\=================== next fold =====================//
                     |
                     v
+-------------------------------------------+
| A7.18  aggregate mean +/- std over folds  |
|   runs OUTSIDE the fold loop              |
+-------------------------------------------+

NOTE the asymmetry: the network is OPTIMISED on cross-entropy (A7.12)
but SELECTED and SCHEDULED on val F1 (A7.15, A7.10).
```

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **A7.1** | Load every `.pt`, cast `x→float`, `edge_index→long`, `edge_attr→float`, reshape `y` to scalar `long`; `try/except` increments `invalid_graphs` | 1,890 files | `list[Data]` | none — deterministic |
| **A7.2** | **`if graph.get("source") == "Li": continue`** | `Data` | filtered list | none — deterministic |
| **A7.3** | `labels = np.array([int(g.y.item())])`; print class histogram | list | `int64[N_A]` | none — deterministic |
| **A7.4** | `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` over `np.zeros(len(labels)), labels` | labels | 10 `(train_val_idx, test_idx)` pairs | none — splitter |
| **A7.5** | `train_test_split(train_val_idx, test_size=0.1, stratify=train_val_labels, random_state=42)` | fold indices | `train_idx`, `val_idx` | none — splitter |
| **A7.6** | `class_weights = total_train / (2 * np.bincount(train_labels, minlength=2))` → CUDA `float32[2]` | train labels | `w ∈ ℝ²` | feeds A7.12 |
| **A7.7** | 3× `DataLoader(batch_size=32)`; `shuffle=True` only for train | 3 lists | 3 loaders, each batch is a disjoint-union graph | none — deterministic |
| **A7.8** | `model = BBB_GNN().to(device)` — **re-instantiated every fold, no weight carry-over** | — | 345,514 params | — |
| **A7.9** | `AdamW(lr=5e-4, weight_decay=5e-4)` — decoupled L2 | params | optimiser state | implements L2 term |
| **A7.10** | `ReduceLROnPlateau(mode="max", factor=0.5, patience=8, min_lr=1e-6)` driven by **val F1**, not val loss | val F1 scalar | LR schedule | none — control |
| **A7.11** | `for epoch in 1..200` | — | — | none — control |
| **A7.12** | **⬡ LOSS NODE** — see §2.6 | logits `[B,2]`, targets `[B]`, weights `[2]` | scalar loss | **weighted softmax cross-entropy** |
| **A7.13** | `loss.backward()`; `clip_grad_norm_(max_norm=2.0)`; `optimizer.step()`; `zero_grad(set_to_none=True)` | scalar loss | updated params | — |
| **A7.14** | `evaluate()` under `no_grad`: `softmax` → `probabilities[:,1]`; argmax preds; accuracy/precision/recall/F1/ROC-AUC with `roc_auc_score` guarded by `try/except ValueError → 0.0` | val loader | 5 metrics | none — measurement |
| **A7.15** | If `val_f1 > best_val_f1`: `torch.save({model_state_dict, node_features:9, edge_features:7, hidden_channels:256, classes:2, best_val_f1, best_epoch})` to `best_bbb_gnn_model_fold_K.pt` | state | ▱ checkpoint | **selection criterion = val F1** |
| **A7.16** | Non-improvement counter; `break` at 20 | — | control | none — control |
| **A7.17** | Reload best checkpoint, evaluate on the fold's untouched `test_idx` | checkpoint + test loader | 5 metrics per fold | none — measurement |
| **A7.18** | `mean ± std` across 10 folds for Acc/Prec/Rec/F1/AUC; print CUDA memory | 10 metric dicts | summary table | none — measurement |

> **Audit flag — A7.2 (draw this prominently).** The `Li` source is filtered out at load time, so
> although **1,890** graphs exist on disk, only the **Adenot subset** is ever trained on. Measured
> from `cleaned_bbb_dataset.csv`: **Adenot = 1,621, Li = 269**. So A7.2 silently discards **14.2%**
> of the curated corpus, and every Paradigm A metric is a 1,621-molecule Adenot-only number. The
> 269 `Li` graphs are featurised, serialised to disk, loaded, and then thrown away — real wasted
> compute, plus a permanent discrepancy between `graph_summary.csv` (1,890 rows) and the training
> log. With 10-fold CV this leaves ~146 molecules per test fold, so per-fold metrics carry a
> standard error of roughly ±3–4 percentage points.

> **Audit flag — A7.8 + A7.15.** Because a fresh model is built per fold and only per-fold
> checkpoints are written, **`train_gnn.py` never produces the un-suffixed `best_bbb_gnn_model.pt`**
> that `FINAL_MODEL`/`BEST_MODEL` name at the top of the file. `predict.py` compensates with a
> fallback to `best_bbb_gnn_model_fold_1.pt`. So the deployed Paradigm A predictor is **fold 1's
> model**, not an ensemble and not a full-data refit. Draw this as a dashed "fallback" arrow.

---

### 2.5 Stage A8 · `BBB_GNN` forward pass, layer by layer

This is the block you most need for hand-drawing. Every tensor shape is explicit.

```text
INPUT   x [V x 9] f32    edge_index [2 x 2E] i64    edge_attr [2E x 7] f32    batch [V] i64
  |
  v
+============================ GINE BLOCK 1 ============================+
| A8.1a  edge encoder   Linear(7 -> 9)                   72 par        |   [2E x 7] -> [2E x 9]
| A8.1b  message        ReLU(x_j + e_ij)                  0 par        |   [2E x 9]
| A8.1c  aggregate      scatter-add                       0 par        |   [V  x 9]
| A8.1d  (1 + eps) * x_i + aggr    eps is a BUFFER        0 par        |   [V  x 9]
| A8.1e  nn1   Linear(9->256) ReLU Linear(256->256)  68,352 par        |   [V x 256]
+===================================+==================================+
     +-----------------------------v------------------------------+
     | A8.2   BatchNorm1d(256)                       512 par      |  [V x 256]
     | A8.3   ReLU                                     0 par      |  [V x 256]
     | A8.4   Dropout(p = 0.30)   train only           0 par      |  [V x 256]
     +-----------------------------+------------------------------+
                                   v
+============================ GINE BLOCK 2 ============================+
| A8.5a  edge encoder   Linear(7 -> 256)              2,048 par        |
| A8.5b  nn2   Linear(256->256) ReLU Linear(256->256)131,584 par       |   [V x 256]
+===================================+==================================+
     +-----------------------------v------------------------------+
     | A8.6 BatchNorm1d | A8.7 ReLU | A8.8 Dropout   512 par      |  [V x 256]
     +-----------------------------+------------------------------+
                                   v
+============================ GINE BLOCK 3 ============================+
| A8.9a  edge encoder   Linear(7 -> 256)              2,048 par        |
| A8.9b  nn3                                         131,584 par       |   [V x 256]
+===================================+==================================+
     +-----------------------------v------------------------------+
     | A8.10 BatchNorm1d | A8.11 ReLU | A8.12 Dropout 512 par     |  [V x 256]
     +-----------------------------+------------------------------+
                                   v
     +############################################################+
     #  A8.13   global_mean_pool(x, batch)                        #  <== ONLY shape-collapsing
     #            [V x 256]   -->   [B x 256]                     #      operation; the atom
     #          segment mean by the batch vector                  #      dimension dies here
     +#############################+##############################+
     +-----------------------------v------------------------------+
     | A8.14  fc1  Linear(256 -> 32)               8,224 par      |  [B x 32]
     | A8.15  ReLU                                     0 par      |  [B x 32]
     | A8.16  Dropout(0.30)                            0 par      |  [B x 32]
     | A8.17  fc2  Linear(32 -> 2)                    66 par      |  [B x 2 ]
     +-----------------------------+------------------------------+
                                   v
                OUTPUT   logits [B x 2] float32   (raw, unnormalised)
                                   |
                                   v
                   (o) A7.12   weighted softmax cross-entropy
                       :
                       +...... gradient flows back through EVERY block above

TOTAL TRAINABLE PARAMETERS = 345,514          RECEPTIVE FIELD = 3 bonds
  conv1  68,352 + 72    =  68,424
  conv2 131,584 + 2,048 = 133,632
  conv3 131,584 + 2,048 = 133,632
  bn1+bn2+bn3           =   1,536
  fc1                   =   8,224
  fc2                   =      66
```

#### A8 shape and parameter ledger

| Block | Operation | IN shape | OUT shape | Params | LOSS |
|---|---|---|---|---|---|
| A8.1a | `GINEConv-1` edge encoder `Linear(7→9)` | `[2E,7]` | `[2E,9]` | 72 | none |
| A8.1b | message `ReLU(x_j + e_ij)` | `[2E,9]` | `[2E,9]` | 0 | none |
| A8.1c | scatter-add aggregation | `[2E,9]` | `[V,9]` | 0 | none |
| A8.1d | `(1+ε)·x_i + aggr`, `train_eps=False` ⇒ ε is a **buffer** | `[V,9]` | `[V,9]` | 0 | none |
| A8.1e | `nn1 = Linear(9→256) → ReLU → Linear(256→256)` | `[V,9]` | `[V,256]` | 68,352 | none |
| A8.2 | `BatchNorm1d(256)` | `[V,256]` | `[V,256]` | 512 | none |
| A8.3 | `ReLU` | `[V,256]` | `[V,256]` | 0 | none |
| A8.4 | `Dropout(0.30)` | `[V,256]` | `[V,256]` | 0 | none |
| A8.5a | `GINEConv-2` edge encoder `Linear(7→256)` | `[2E,7]` | `[2E,256]` | 2,048 | none |
| A8.5b | `nn2` | `[V,256]` | `[V,256]` | 131,584 | none |
| A8.6–A8.8 | `BatchNorm1d` → `ReLU` → `Dropout` | `[V,256]` | `[V,256]` | 512 | none |
| A8.9a | `GINEConv-3` edge encoder `Linear(7→256)` | `[2E,7]` | `[2E,256]` | 2,048 | none |
| A8.9b | `nn3` | `[V,256]` | `[V,256]` | 131,584 | none |
| A8.10–A8.12 | `BatchNorm1d` → `ReLU` → `Dropout` | `[V,256]` | `[V,256]` | 512 | none |
| A8.13 | `global_mean_pool(x, batch)` | `[V,256]` | `[B,256]` | 0 | none |
| A8.14 | `fc1 = Linear(256→32)` | `[B,256]` | `[B,32]` | 8,224 | none |
| A8.15–A8.16 | `ReLU` → `Dropout(0.30)` | `[B,32]` | `[B,32]` | 0 | none |
| A8.17 | `fc2 = Linear(32→2)` | `[B,32]` | `[B,2]` | 66 | feeds ⬡ A7.12 |

**Total learnable parameters = 345,514.**

```
  conv1 : 68,352 (nn1) +     72 (edge encoder)  =  68,424
  conv2 : 131,584 (nn2) +  2,048 (edge encoder) = 133,632
  conv3 : 131,584 (nn3) +  2,048 (edge encoder) = 133,632
  bn1+bn2+bn3 : 3 x 512                         =   1,536
  fc1                                           =   8,224
  fc2                                           =      66
  ----------------------------------------------------------
  TOTAL                                         = 345,514
```

> **Note on the parameter count — this figure is derived, not executed.** `torch_geometric` is not
> installed in the environment this audit ran in, so the total could not be confirmed empirically.
> The **345,514** above is an itemised hand census from the layer definitions, and it disagrees with
> the 344,023 stated in `ARCHITECTURE_DEEP_DIVE.md` by 1,491. The likely cause is the three
> `GINEConv` edge-encoder `Linear` layers (72 + 2,048 + 2,048 = 4,168 params), which PyG creates
> implicitly when `edge_dim` is passed and which are easy to omit when counting by hand. Confirm in
> one line before quoting either number:
>
> ```python
> sum(p.numel() for p in BBB_GNN().parameters() if p.requires_grad)
> ```
>
> One caveat that affects the census: with `train_eps=False` (the default, and what this code uses)
> PyG registers `GINEConv.eps` as a **buffer**, not a `Parameter`, so it contributes 0 to the
> trainable count. If you ever set `train_eps=True`, add 1 parameter per conv layer.

> **Receptive field.** Three GINE layers ⇒ each atom's final embedding summarises its **3-bond
> neighbourhood**. Draw three concentric dashed circles around one atom to show this. It is the
> structural reason the model captures local functional groups well but cannot see
> whole-molecule globular shape — precisely the gap Paradigm B fills with explicit 3D descriptors.

---

### 2.6 ⬡ A7.12 · The Paradigm A loss function, stated exactly

`train_gnn.py:356` — `F.cross_entropy(output, batch.y, weight=class_weights)`

**Weighted softmax cross-entropy over 2 classes.** PyTorch fuses `log_softmax` + `nll_loss`, so the
raw logits from A8.17 are consumed directly and no explicit softmax exists in the training graph.

```
                     exp( z_{i, y_i} )
  p_{i, y_i}  =  -----------------------
                   SUM_c exp( z_{i, c} )

                 SUM_i  w_{y_i} * ( - log p_{i, y_i} )
  L_CE      =   -----------------------------------        ( reduction = "mean" )
                            SUM_i  w_{y_i}

                      N_train
  w_c       =   -----------------                           ( inverse-frequency balancing )
                   2 * n_c
```

| Element | Value in this repo |
|---|---|
| Logits `z` | `[B,2]` from A8.17 |
| Targets `y` | `[B]` `int64 ∈ {0,1}` |
| Class weights `w` | `float32[2]`, computed per fold in A7.6 |
| Reduction | `"mean"` — **weighted** mean, denominator is `Σ w_{y_i}`, not `B` |
| Regularisation | **not in `L_CE`** — L2 enters via `AdamW(weight_decay=5e-4)` as a decoupled update, so the reported loss value excludes it |
| Gradient control | `clip_grad_norm_(max_norm=2.0)` on the global parameter-vector norm |
| Model selection | **val F1**, not `L_CE` (A7.15) |
| LR schedule driver | **val F1** maximisation (A7.10) |

> **Draw the loss node as a hexagon with three inbound arrows** (logits, targets, class weights) and
> **two outbound dashed arrows**: one gradient arrow back to A8, and one scalar arrow to the
> epoch-logging block. Then draw a *separate* dashed arrow from A7.14's val-F1 output to both A7.10
> and A7.15 — this visually encodes the key asymmetry: **the network is optimised on
> cross-entropy but selected and scheduled on F1.**

---

### 2.7 Stage A10 · Paradigm A inference path

Source: `Initial Model/predict.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **A10.1** | Model-file resolution: `best_bbb_gnn_model.pt`, else fallback `best_bbb_gnn_model_fold_1.pt` | — | path | none — control |
| **A10.2** | Re-declare `BBB_GNN` **without dropout in `forward`** | — | module | none — architecture |
| **A10.3** | `load_state_dict(checkpoint["model_state_dict"])`; `model.eval()` | ▱ checkpoint | weights loaded | none — deterministic |
| **A10.4** | `get_atom_features` using `atom.GetValence(Chem.ValenceType.EXPLICIT/IMPLICIT)` (modern RDKit API) | `Atom` | `list[9]` | none — deterministic |
| **A10.5** | `get_bond_features` — identical 7-vector to A4.3 | `Bond` | `list[7]` | none — deterministic |
| **A10.6** | `molecule_to_graph`: builds `x`, bidirectional `edge_index`, `edge_attr`, plus **`batch = zeros(V)`**; empty-molecule guard emits `edge_index [2,0]`, `edge_attr [0,7]` | SMILES | `Data` | none — deterministic |
| **A10.7** | `forward` with `torch.mean(x, dim=0, keepdim=True)` substituted for `global_mean_pool` | `[V,256]` | `[1,256]` | none — deterministic |
| **A10.8** | `F.softmax(output, dim=1)`; `argmax`; `confidence = max(p0,p1)` | `[1,2]` logits | class, confidence, `p0`, `p1` | none — deterministic |
| **A10.9** | `while True` REPL; prints prediction, probabilities, node/edge counts, device; `exit`/`quit`/`q` to break | stdin | stdout | none — I/O |

> **Three inference-path discrepancies worth annotating on the drawing.**
> 1. **Pooling substitution (A10.7).** `torch.mean(x, dim=0, keepdim=True)` is numerically identical
>    to `global_mean_pool` *only* because `batch` is all zeros (single molecule). It would silently
>    produce one averaged vector for a whole mini-batch if ever used for batched inference.
> 2. **Dropout absence (A10.2).** Harmless, because `model.eval()` would disable it anyway, but it
>    means the inference class is **not** the training class — a latent drift risk.
> 3. **BatchNorm in eval mode.** A8.2/A8.6/A8.10 use running statistics accumulated during training.
>    With `V` as the effective batch dimension, these running stats were estimated over *atoms
>    pooled across molecules*, so single-molecule inference is statistically consistent only to the
>    extent that one molecule's atom distribution resembles the training corpus.

---

## SECTION 3 — PARADIGM B: SHARED FEATURE KERNEL AND PREPROCESSING

### 3.1 Stage B3 · `core/features.py` — the shared feature kernel

This module is the **single source of truth** for featurisation. Preprocessing (B1, B2), inference
(B9), screening (D4), and tests (D6) all route through it. Draw it as one central reusable block
with multiple callers.

```text
                                 INPUT * SMILES string
                                          |
                                          v
                           +------------------------------+
                           | B3.1  MolFromSmiles          |
                           |   None  ==>  abort           |
                           +---+----------+-----------+---+
                               |          |           |
                   +-----------v--+  +----v-------+   |
                   | B3.4  Morgan |  | B3.2  2D   |   |
                   |   radius 2   |  |   descrip- |   |
                   |   2048 bits  |  |   tors x6  |   |
                   |   cached gen |  +----+-------+   |
                   +------+-------+       |           |
                          |               |           v
                          |               |   +------------------------------+
                          |               |   | B3.3  AddHs + ETKDGv3        |
                          |               |   |   randomSeed = 42            |
                          |               |   +--------------+---------------+
                          |               |                  v
                          |               |          <> embed ok ?
                          |               |          |
                          |               |          +-- no --> B3.3b retry
                          |               |          |          maxAttempts = 100
                          |               |          |          |
                          |               |          |          +-- no --> (None, None)
                          |               |          |          |          ABORT: B3.6
                          |               |          |          |          returns None
                          |               |          |          +-- yes --+
                          |               |          +-- yes --------------+
                          |               |                                v
                          |               |              +------------------------------+
                          |               |              | B3.3c  Descriptors3D x6      |
                          |               |              |  Rg Asph Ecc Spher NPR1 NPR2 |
                          |               |              +--------------+---------------+
                          |               |                             |
          +---------------+---------------+                             |
          |  fp[2048] + desc_2d[6]        |  fp[2048] + desc_2d[6]      | desc_3d[6]
          |                               |                             |
          v                               v                             v
+----------------------------------+    +-------------------------------------------+
| B3.5  extract_features_bbb       |    | B3.6  extract_features_pgp                |
|   returns                        |    |   returns                                 |
|   (fp[2048], desc[6]), raw, mol  |    |   (fp[2048], desc[12]), raw, mol_h        |
|                                  |    |   desc order = 2D x6 THEN 3D x6           |
+----------------+-----------------+    +---------------------+---------------------+
                 |                                            |
                 |  6 continuous                              |  12 continuous
                 +---------------------+----------------------+
                                       v
                     +-----------------------------------------------+
  (raw) scaler JSON  | B3.7  prepare_dataframe                       |
  ------------------>|   z = (v - mean) / (std if std != 0 else 1.0) |
  B3.8 load_scaler   |   applies to continuous cols ONLY --          |
       metadata      |   the 2,048 binary FP cols are NEVER scaled   |
                     |                                               |
  BBB call passes    |   COLUMN ORDER IS FIXED AND LOAD-BEARING:     |
  hard-coded         |     FP_0 .. FP_2047, then named descriptors   |
  defaults;          +-----------------------+-----------------------+
  P-gp call does not                         v
                        OUTPUT   DataFrame [1 x 2054]   (BBB)
                                 DataFrame [1 x 2060]   (P-gp)

TWO SEPARATE SCALERS -- never cross them.
     BBB   mean MolWt = 383.86    mean LogP = 2.32
     P-gp  mean MolWt = 377.03    mean LogP = 3.41
Draw two distinct cylinders. Using the wrong one silently shifts every
continuous input and every downstream tree split.
```

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **B3.1** | `Chem.MolFromSmiles(smiles)`; `RDLogger.DisableLog('rdApp.*')` globally | SMILES str | `Mol` or `None` | none — deterministic |
| **B3.2** | `calculate_2d_descriptors` → dict of 6 floats (see table below) | `Mol` | `dict[6]` | none — deterministic |
| **B3.3** | `Chem.AddHs(mol)`; `AllChem.ETKDGv3()` with `randomSeed=42`; `EmbedMolecule` | `Mol` | `mol_h` with conformer, or `None` | none — stochastic-but-seeded |
| **B3.3b** | On `res != 0`: retry `EmbedMolecule(randomSeed=42, maxAttempts=100)`; on second failure return `(None, None)` | `Mol` | `mol_h` or `None` | none — control |
| **B3.3c** | 6 × `Descriptors3D.*` on the embedded conformer | `mol_h` | `dict[6]` | none — deterministic |
| **B3.4** | Module-level cached `_MORGAN_GEN_R2_2048 = GetMorganGenerator(radius=2, fpSize=2048)`; `ConvertToNumpyArray` into `int8` then `.astype(int)` | `Mol` | `int[2048]` bit array | none — deterministic |
| **B3.5** | `extract_features_bbb` → `((fp[2048], desc[6]), raw_dict, mol)` | SMILES | tuple | none — deterministic |
| **B3.6** | `extract_features_pgp` → `((fp[2048], desc[12]), raw_dict, mol_h)`; **returns `(None, None, mol)` if 3D embedding failed** | SMILES | tuple | none — deterministic |
| **B3.7** | `prepare_dataframe`: `scaled[i] = (v - means[i]) / (stds[i] or 1.0)`; builds `FP_0…FP_2047` then named descriptors, in that exact column order | fp + desc + means + stds | `DataFrame [1 × 2054/2060]` | none — deterministic |
| **B3.8** | `load_scaler_metadata` — reads JSON `{means, stds, features}`; falls back to hard-coded defaults if absent/corrupt | path | `(means, stds)` | none — I/O |

#### B3.2 · The 6 shared 2D descriptors (`BBB_DESCRIPTOR_NAMES`)

| Order | Name | RDKit call | Physical meaning | Train mean (BBB) | Train std (BBB) |
|---|---|---|---|---|---|
| 0 | `MolWt` | `Descriptors.MolWt` | molecular mass, Da | 383.860 | 168.403 |
| 1 | `LogP` | `Descriptors.MolLogP` | Crippen lipophilicity | 2.320 | 2.189 |
| 2 | `TPSA` | `Descriptors.TPSA` | topological polar surface area, Å² | 87.097 | 67.043 |
| 3 | `HDonors` | `Descriptors.NumHDonors` | H-bond donor count | 2.021 | 2.240 |
| 4 | `HAcceptors` | `Descriptors.NumHAcceptors` | H-bond acceptor count | 5.299 | 3.626 |
| 5 | `RotBonds` | `Descriptors.NumRotatableBonds` | conformational flexibility | 4.783 | 3.672 |

#### B3.3c · The 6 additional 3D shape descriptors (P-gp only)

| Order | Name | Physical meaning | Train mean (P-gp) | Train std (P-gp) |
|---|---|---|---|---|
| 6 | `RadiusOfGyration` | mass-weighted spatial extent, Å | 4.455 | 1.143 |
| 7 | `Asphericity` | deviation from spherical symmetry, 0–1 | 0.474 | 0.185 |
| 8 | `Eccentricity` | elongation of the inertia ellipsoid | 0.960 | 0.043 |
| 9 | `SpherocityIndex` | sphere-likeness, 0–1 | 0.187 | 0.111 |
| 10 | `NPR1` | normalised PMI ratio `I1/I3` | 0.246 | 0.125 |
| 11 | `NPR2` | normalised PMI ratio `I2/I3` | 0.849 | 0.100 |

> **Note `PGP_DESCRIPTOR_NAMES` ordering is 2D-then-3D** — the 6 shared 2D descriptors occupy
> indices 0–5 and must stay there, because `prepare_dataframe` and the P-gp scaler metadata
> both index positionally. The P-gp scaler means for `MolWt` (377.03) and `LogP` (3.41) differ from
> the BBB scaler (383.86, 2.32), which is correct — **each model has its own independently fitted
> scaler**, and crossing them would silently shift every input. Draw two distinct scaler cylinders.

---

### 3.2 Stage B1 · BBB preprocessing

Source: `B3DB Based Model/BBB/bbb_data_preprocessing.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **B1.1** | `pd.read_csv(sep='\t')` on `B3DB_classification.tsv` | ▱ TSV, 7,809 rows | DataFrame | none — deterministic |
| **B1.2** | Project `['SMILES','BBB+/BBB-']`, `dropna` | DF | DF | none — deterministic |
| **B1.3** | `Label = {'BBB+':1, 'BBB-':0}`; drop original column | DF | DF with `int` label | none — deterministic |
| **B1.4** | Row loop: `MolFromSmiles`; if valid → `calculate_morgan_fingerprint` (B3.4) + `calculate_2d_descriptors` (B3.2); collect `valid_indices` | DF | 3 parallel lists | none — deterministic |
| **B1.5** | Build `fp_df [N × 2048]` with columns `FP_0…FP_2047`, `desc_df [N × 6]`, both re-indexed to `df_valid.index` | lists | 2 DFs | none — deterministic |
| **B1.6** | `pd.concat([df_valid, fp_df, desc_df], axis=1)` → `to_csv` | DFs | ⬓ `processed_b3db_data_advanced.csv` `[7805 × 2056]` | none — I/O |

> **Critical property: B1 writes RAW, UNSCALED descriptors.** Standardisation is deliberately
> deferred to B4.3 so the scaler can be fitted on the training partition only. The module declares
> `SCALER_METADATA_PATH` and imports `StandardScaler` but **never uses either** — dead code.
> Draw B1's output cylinder labelled **"raw descriptors"** in bold; this is the hinge that makes the
> whole pipeline leakage-free, and misreading it is the most likely way to misdraw Paradigm B.

---

### 3.3 Stage B2 · P-gp preprocessing

Source: `B3DB Based Model/Pgp/pgp_data_preprocessing.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **B2.1** | `pd.read_csv(sep='\t')` on `pgp_broccatelli.tab` | ▱ 1,219 rows | DataFrame (`Drug`, `Y`) | none — deterministic |
| **B2.2** | `tqdm` row loop: `MolFromSmiles` on column `Drug` | DF | `Mol` | none — deterministic |
| **B2.3** | `calculate_3d_descriptors(mol, seed=42)` (B3.3) — **gate: skip molecule entirely if `desc_3d is None`** | `Mol` | `dict[6]` or skip | none — control |
| **B2.4** | `calculate_morgan_fingerprint` + `calculate_2d_descriptors`; merge into one 12-key dict via `desc_2d.update(desc_3d)` | `Mol` | `int[2048]`, `dict[12]` | none — deterministic |
| **B2.5** | Rename `Drug→SMILES`, `Y→Label`; project to those 2 columns | DF | DF `[N × 2]` | none — deterministic |
| **B2.6** | Concat with `fp_df [N × 2048]` and `desc_df [N × 12]` → `to_csv` | DFs | ⬓ `processed_pgp_data.csv` `[1205 × 2062]` | none — I/O |

> **The 3D conformer gate (B2.3) is a lossy filter: 1,219 → 1,205 molecules (14 dropped, 1.1%).**
> Draw it as a diamond with a labelled reject path. This is the price Paradigm B pays for 3D shape
> features, and it is the reason the P-gp and BBB datasets cannot be joined row-wise.

> **Note on `desc_2d.update(desc_3d)`.** This mutates the dict returned by `calculate_2d_descriptors`
> in place and then aliases it as `combined_desc`. It is safe here only because B3.2 constructs a
> fresh dict on every call. Worth a small annotation as a latent aliasing hazard.

---

## SECTION 4 — PARADIGM B: MODEL TRAINING

### 4.1 Stage B4 · BBB model — nested Optuna + XGBoost

Source: `B3DB Based Model/BBB/bbb_model_training.py`

```text
   (db) processed_b3db_data_advanced.csv   [7,805 x 2,056]   RAW descriptors
            |
            v
+----------------------------------------------+
| B4.2a  drop SMILES, Label                    |
|   X [7805 x 2054]      y [7805]              |
+---------------------+------------------------+
                      v
+----------------------------------------------+        +=====================+
| B4.2b  SPLIT 1   test_size=0.15, stratified  |  ----> |  X_test  [1,171]    |
|   X_temp [6,634]                             |        |  ** PADLOCKED **    |
+---------------------+------------------------+        |  touched EXACTLY    |
                      v                                 |  once, at B4.6      |
+----------------------------------------------+        +==========+==========+
| B4.2c  SPLIT 2   val_ratio = 0.15/0.85       |                   |
|   X_train [5,463]      X_val [1,171]         |                   |
+---------------------+------------------------+                   |
                      v                                            |
+----------------------------------------------+                   |
| B4.3a  StandardScaler   FIT ON TRAIN ONLY    |                   |
|   6 continuous cols only                     |                   |
|   the 2,048 binary FP cols are NEVER scaled  |                   |
+------+----------------------+----------------+                   |
       v                      v                                    |
+--------------+   +----------------------------+                  |
| B4.3b        |   | B4.3c  dump means / stds   |                  |
|  transform   |   |  (raw) bbb_scaler_         |                  |
|  val + test  |   |        metadata.json       |                  |
+------+-------+   +----------------------------+                  |
       v                                                           |
+----------------------------------------------+                   |
| B4.3d  get_optimal_device (cuda probe)       |                   |
| B4.3e  scale_pos_weight = n_neg / n_pos      |                   |
|          ~ 0.575 => positives DOWN-weighted  |                   |
+---------------------+------------------------+                   |
                      v                                            |
+==============================================================+   |
| (o) B4.4   OUTER SEARCH OBJECTIVE                            |   |
|       maximise F1 on X_val    *   Optuna TPE, 50 trials      |   |
|       BLACK-BOX, NON-DIFFERENTIABLE                          |   |
|  +--------------------------------------------------------+  |   |
|  | B4.4a  inner fit   n_estimators = 1000                 |  |   |
|  |        early_stopping_rounds = 30, eval_set = X_val    |  |   |
|  |   +--------------------------------------------------+ |  |   |
|  |   | (o) B4.4b   INNER LOSS                           | |  |   |
|  |   |   binary:logistic logloss + Omega                | |  |   |
|  |   |   g_i = w*(p - y)      h_i = w*p*(1 - p)         | |  |   |
|  |   |   DIFFERENTIABLE, SECOND-ORDER                   | |  |   |
|  |   +--------------------------------------------------+ |  |   |
|  +--------------------------+-----------------------------+  |   |
+=============================+================================+   |
                              v                                    |
+----------------------------------------------+                   |
| B4.4c  study.best_params   (9 hyperparams)   |                   |
+---------------------+------------------------+                   |
                      v                                            |
+----------------------------------------------+                   |
| B4.5   FINAL REFIT   n_estimators = 2000     |                   |
|        early_stopping_rounds = 50            |                   |
|   (o) B4.5a  same loss form as B4.4b         |                   |
+---------------------+------------------------+                   |
                      v                                            |
+----------------------------------------------+                   |
| B4.6   LOCKED-VAULT TEST EVAL                | <-----------------+
|   acc  bacc  prec  rec  F1  ROC-AUC          |   THE ONLY READ
+---------------------+------------------------+   OF X_test
                      v
+----------------------------------------------+
| B4.7  (db) bbb_xgboost_optuna_best.json      |
|   494 trees * 2,054 features                 |
|   best_iteration = 443  (443 + 50 + 1 = 494) |
+----------------------------------------------+

SUBTLETY: early stopping in B4.5 watches X_val, which Optuna already maximised
F1 over. X_val is therefore OPTIMISTICALLY BIASED. Only X_test is clean --
which is exactly why run_validation.py (D1) re-derives this same 15% split.
```

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **B4.1** | `read_csv(processed_b3db_data_advanced.csv)` | ⬓ CSV | DF `[7805 × 2056]` | none — I/O |
| **B4.2a** | `X = df.drop(columns=['SMILES','Label'])`, `y = df['Label']` | DF | `X [7805×2054]`, `y [7805]` | none — deterministic |
| **B4.2b** | `train_test_split(test_size=0.15, random_state=42, stratify=y)` | `X,y` | `X_temp [6634]`, `X_test [1171]` | none — splitter |
| **B4.2c** | `val_ratio = 0.15/0.85 = 0.17647`; `train_test_split(X_temp, test_size=val_ratio, stratify=y_temp)` | `X_temp` | `X_train [5463]`, `X_val [1171]` | none — splitter |
| **B4.3a** | `StandardScaler().fit_transform(X_train[BBB_DESCRIPTOR_NAMES])` — **6 continuous columns only; the 2,048 binary FP columns are never scaled** | `X_train` | scaled train | none — deterministic |
| **B4.3b** | `scaler.transform` applied to `X_val` and `X_test` | val, test | scaled | none — deterministic |
| **B4.3c** | `json.dump({means, stds, features})` → `bbb_scaler_metadata.json` | scaler state | ▱ JSON | none — I/O |
| **B4.3d** | `get_optimal_device()` — fits a throwaway 2-estimator `device='cuda'` model on a 5×2 dummy array inside `try/except`; returns `'cuda'` or `'cpu'` | — | device str | none — probe |
| **B4.3e** | `scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)` | train labels | scalar | feeds both loss nodes |
| **B4.4** | **⬡ OUTER SEARCH OBJECTIVE** — `optuna.create_study(direction="maximize")`, `n_trials=50`, TPE sampler (default) | search space | `best_params` | **maximise F1 on `X_val`** (non-differentiable) |
| **B4.4a** | Per trial: `XGBClassifier(n_estimators=1000, early_stopping_rounds=30, eval_metric='logloss', tree_method='hist')` fitted on train with `eval_set=[(X_val,y_val)]` | train + val | fitted booster | see B4.4b |
| **B4.4b** | **⬡ INNER TRAINING LOSS** — `binary:logistic` logloss + structural penalties | — | gradients/hessians | **weighted binary cross-entropy + Ω** |
| **B4.4c** | `study.best_params`, `study.best_value` printed via `print_best_callback` | 50 trials | 9 hyperparameters | none — selection |
| **B4.5** | Final refit with `best_params` overridden: `n_estimators=2000`, `early_stopping_rounds=50`, `eval_set=[(train),(val)]`, `verbose=100` | train + val | deployed booster | see B4.5a |
| **B4.5a** | **⬡ FINAL TRAINING LOSS** — identical functional form to B4.4b | — | trees | **weighted binary cross-entropy + Ω** |
| **B4.6** | Test-set scoring: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC | `X_test` | 6 scalars | none — measurement |
| **B4.7** | `save_model(bbb_xgboost_optuna_best.json)` | booster | ⬓ **494 trees, 2,054 features, `best_iteration=443`, `base_score=0.5`** | none — I/O |

#### B4.4 · Optuna search space, exactly as coded

| Hyperparameter | Distribution | Range |
|---|---|---|
| `learning_rate` | `suggest_float(log=True)` | 0.01 – 0.3 |
| `max_depth` | `suggest_int` | 3 – 10 |
| `subsample` | `suggest_float` | 0.5 – 1.0 |
| `colsample_bytree` | `suggest_float` | 0.5 – 1.0 |
| `min_child_weight` | `suggest_int` | 1 – 10 |
| `gamma` | `suggest_float` | 0.0 – 5.0 |
| `reg_alpha` | `suggest_float(log=True)` | 1e-3 – 10.0 |
| `reg_lambda` | `suggest_float(log=True)` | 1e-3 – 10.0 |
| *fixed* | `scale_pos_weight` | from B4.3e |

> **Two-level optimisation — draw as two nested boxes.** The inner loop (B4.4b) minimises a smooth,
> differentiable surrogate (logloss) by gradient boosting. The outer loop (B4.4) maximises a
> non-differentiable, threshold-dependent metric (F1 at 0.5) by Bayesian search. The outer objective
> cannot be back-propagated — that is *why* Optuna exists in this pipeline. Label the inner box
> "differentiable, 2nd-order" and the outer box "black-box, TPE".

> **Leakage containment — draw `X_test` as a padlocked cylinder.** `X_test` is created in B4.2b and
> then touched exactly once, at B4.6. It is absent from B4.3a (scaler fit), from every Optuna trial,
> and from the early-stopping `eval_set`. The one subtlety: **early stopping in B4.5 monitors
> `X_val`, which Optuna already maximised F1 over**, so `X_val` is optimistically biased and only
> `X_test` is a clean estimate. That is exactly why `run_validation.py` (D1) re-derives the identical
> 15% test partition rather than trusting the validation numbers.

---

### 4.2 Stage B5 · P-gp model — single fixed-hyperparameter fit

Source: `B3DB Based Model/Pgp/pgp_model_training.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **B5.1** | `read_csv(processed_pgp_data.csv)`; print shape and `y.value_counts()` | ⬓ CSV | DF `[1205 × 2062]` | none — I/O |
| **B5.2** | `X = drop(['SMILES','Label'])`, `y = df['Label']` | DF | `X [1205×2060]`, `y` | none — deterministic |
| **B5.3** | `train_test_split(test_size=0.2, random_state=42, stratify=y)` → train 964 / test 241 | `X,y` | 4 partitions | none — splitter |
| **B5.4a** | `StandardScaler().fit_transform(X_train[PGP_DESCRIPTOR_NAMES])` — **12 continuous columns only** | `X_train` | scaled | none — deterministic |
| **B5.4b** | `transform(X_test)`; `json.dump` → `pgp_scaler_metadata.json` | test | scaled + ▱ JSON | none — I/O |
| **B5.4c** | `scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)` | labels | scalar | feeds B5.5 |
| **B5.5** | **⬡ TRAINING LOSS** — `XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight, eval_metric='logloss', tree_method='hist', random_state=42)` fitted with `eval_set=[(train),(test)]`, `verbose=50` | train | 300 trees | **weighted binary cross-entropy** (no `gamma`/`alpha`/`lambda` overrides ⇒ XGBoost defaults `γ=0, α=0, λ=1`) |
| **B5.6** | Test metrics: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC | test | 6 scalars | none — measurement |
| **B5.7** | `save_model(pgp_xgboost_model.json)` | booster | ⬓ **300 trees, 2,060 features, `base_score=0.5`** | none — I/O |

> **Asymmetry between the two models — annotate it.** BBB gets 50 Optuna trials + early stopping and
> converges to 494 trees (stopping at iteration 443 of a 2,000 budget). P-gp gets **no tuning and no
> early stopping**: all 300 trees are built unconditionally. Draw B5 as a visibly simpler box than
> B4 with a note "untuned baseline".

> **Audit flag — B5.5 passes the test set as an `eval_set` member.** `eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)]`
> with no `early_stopping_rounds`. Because nothing acts on the watchlist, **no information flows from
> test into the fitted trees** — the test metrics in B5.6 remain valid. But it is one keyword away
> from genuine leakage: adding `early_stopping_rounds` here would select tree count on the test set.
> Draw this as a dashed amber arrow from `X_test` into B5.5 labelled **"monitored only, never
> optimised — fragile"**.

---

### 4.3 ⬡ B4.4b / B4.5a / B5.5 · The Paradigm B loss function, stated exactly

All three XGBoost blocks share one objective: `binary:logistic`, confirmed inside both saved model
JSONs (`learner.objective.name = "binary:logistic"`, `base_score = 0.5`).

**Per-instance loss (weighted binary cross-entropy / logloss):**

```
  F(x_i)  =  SUM over k of  f_k(x_i)        additive ensemble of K regression trees

  p_i     =  sigmoid( F(x_i) )  =  1 / ( 1 + exp( -F(x_i) ) )

  loss(y_i, p_i)  =  - omega_i * [  y_i * log(p_i)  +  (1 - y_i) * log(1 - p_i)  ]

                      { scale_pos_weight    if  y_i = 1
  omega_i         =   {
                      { 1                   if  y_i = 0
```

**Regularised objective minimised at boosting round `t`:**

```
  Obj^(t)   =   SUM over i of  loss( y_i , sigmoid( F^(t-1)(x_i) + f_t(x_i) ) )
                                                               +   Omega( f_t )

  Omega(f)  =     gamma  * T
                + lambda * 0.5 * SUM over j = 1..T of ( w_j )^2
                + alpha  *       SUM over j = 1..T of | w_j |

      where   T    =  number of leaves in tree f
              w_j  =  weight (output value) of leaf j
```

XGBoost expands this to second order with the exact gradient/Hessian pair:

```
  gradient   g_i  =  d(loss) / dF        =  omega_i * ( p_i - y_i )

  hessian    h_i  =  d^2(loss) / dF^2    =  omega_i * p_i * ( 1 - p_i )

  For the leaf j that holds instance set I_j:

                                      ( SUM over i in I_j of g_i )^2
      split gain   is proportional to  -------------------------------
                                      ( SUM over i in I_j of h_i ) + lambda

                                      - ( SUM over i in I_j of g_i )
      optimal leaf weight    w_j*  =   -------------------------------
                                      ( SUM over i in I_j of h_i ) + lambda
```

| Symbol | BBB model (B4) | P-gp model (B5) |
|---|---|---|
| Final `K` (trees) | **494** (early-stopped at iteration 443 of 2,000) | **300** (full budget, no early stopping) |
| `d` (features) | **2,054** | **2,060** |
| `base_score` | 0.5 | 0.5 |
| `γ` (`gamma`) | Optuna, 0.0–5.0 | default **0** |
| `α` (`reg_alpha`) | Optuna, 1e-3–10 log | default **0** |
| `λ` (`reg_lambda`) | Optuna, 1e-3–10 log | default **1** |
| `η` (`learning_rate`) | Optuna, 0.01–0.3 log | fixed **0.05** |
| `max_depth` | Optuna, 3–10 | fixed **5** |
| `subsample` / `colsample_bytree` | Optuna, 0.5–1.0 each | fixed **0.8** / **0.8** |
| `min_child_weight` | Optuna, 1–10 | default **1** |
| `scale_pos_weight` | `n_neg/n_pos` of train | `n_neg/n_pos` of train |
| `eval_metric` | `logloss` | `logloss` |
| `tree_method` | `hist` | `hist` |
| Early stopping | 30 (tuning) / 50 (final) | **none** |
| Model-selection metric | **F1 on val** (Optuna) | none |

> **Drawing instruction for the loss hexagon.** Draw ⬡ with four inbound arrows — `y` (labels),
> `F(x)` (current ensemble margin), `ω` (`scale_pos_weight`), and `Ω` parameters (`γ, α, λ`) — and
> two outbound arrows labelled `g_i` and `h_i` feeding the tree-splitting block. This makes visible
> the one thing that distinguishes boosting from the GNN: **the loss is consumed as a
> gradient/Hessian *pair*, not a single gradient.**

#### Measured class balance, and what `scale_pos_weight` actually does here

| Dataset | Label 0 | Label 1 | Positive rate | `scale_pos_weight` ≈ `n_neg/n_pos` |
|---|---|---|---|---|
| BBB (`processed_b3db`, 7,805) | 2,849 `BBB-` | **4,956 `BBB+`** | **63.5%** | **≈ 0.575** |
| P-gp (`processed_pgp`, 1,205) | 555 non-substrate | **650 substrate** | **53.9%** | **≈ 0.854** |

> **Note the direction of the correction, because it is counter-intuitive.** In both datasets the
> **positive** class is the majority, so `scale_pos_weight < 1` and the weight **down-weights
> positives** rather than boosting a rare class. For BBB this is a substantial correction (0.575),
> and it is the mechanism that keeps the model from simply predicting `BBB+` everywhere. It is also
> why the scaffold-split specificity collapse in §6.7 is diagnostic rather than cosmetic: under
> distribution shift the model reverts to the majority-class behaviour that the loss weighting was
> installed to suppress. Annotate the ⬡ hexagon with `ω₁ = 0.575` (BBB) and `ω₁ = 0.854` (P-gp).

> **Why `scale_pos_weight` and not resampling.** Both datasets are label-imbalanced. Re-weighting the
> loss keeps every molecule in the training set (important at only 964 P-gp training molecules) while
> re-balancing the gradient contribution. Note it rescales `g_i` **and** `h_i`, so it shifts leaf
> values but leaves `base_score` at 0.5 — predicted probabilities are therefore **not calibrated**,
> which is exactly why D1 reports Brier score alongside ROC-AUC.

---

## SECTION 5 — PARADIGM B: INFERENCE AND DECISION

### 5.1 Stage B9 + B8 · Inference orchestration and the four-quadrant matrix

Source: `B3DB Based Model/inference.py`, `core/decision_matrix.py`

```text
+--------------------------------------------------------------+
| B9.1  argparse                                               |
|   --smiles --explain --counterfactual --test-benchmark       |
|   --export-html --export-csv --export-json                   |
+---------------------------+----------------------------------+
                            v
+--------------------------------------------------------------+
| B9.2  load bbb_xgboost_optuna_best.json                      |
| B9.3  load pgp_xgboost_model.json                            |
| B9.4  load_scaler_metadata x2                                |
| B9.5  init C-layer: SHAP, Mapper, SharedProps, Domain, CF    |
+---------------------------+----------------------------------+
                            v
                   <> B9.6  branch
                   |
                   +-- --test-benchmark --> B9.17  5 benchmark drugs
                   +-- --smiles ----------> single shot
                   +-- (none) ------------> interactive REPL
                            |
              +-------------+-------------+
              v                           v
+--------------------------+  +--------------------------+
| B9.7  extract_features_  |  | B9.8  extract_features_  |
|       bbb                |  |       pgp                |
+------------+-------------+  +------------+-------------+
             +--------------+--------------+
                            v
                   <> B9.9  validity gate
                   |
                   +-- either is None --> ERROR, sys.exit(1)
                   +-- both ok
                            |
              +-------------+-------------+
              v                           v
+--------------------------+  +--------------------------+
| B9.10 prepare_dataframe  |  | B9.11 prepare_dataframe  |
|        [1 x 2054]        |  |        [1 x 2060]        |
+------------+-------------+  +------------+-------------+
             v                              v
+--------------------------+  +--------------------------+
| B9.12 predict_proba[:,1] |  | B9.13 predict_proba[:,1] |
|        -> P(BBB+)        |  |        -> P(P-gp+)       |
+------------+-------------+  +------------+-------------+
             +--------------+--------------+
                            v
+--------------------------------------------------------------+
| B9.14  threshold:  pred = int(prob > 0.5)   -- BOTH barriers  |
+---------------------------+----------------------------------+
                            v
+--------------------------------------------------------------+
| B8.1   determine_dual_barrier_quadrant(bbb_pred, pgp_pred)   |
|                                                              |
|          P(P-gp+)                                            |
|      1.0 +-------------------+-------------------+            |
|          |  Q4               |  Q2               |            |
|          |  BBB- / P-gp+     |  BBB+ / P-gp+     |            |
|          |  RESTRICTED       |  PARADOXICAL      |            |
|          |  effluxed         |  Loperamide       |            |
|      0.5 +-------------------+-------------------+            |
|          |  Q3               |  Q1               |            |
|          |  BBB- / P-gp-     |  BBB+ / P-gp-     |            |
|          |  PERIPHERAL       |  OPTIMAL CNS      |            |
|          |  Penicillin G     |  Levetiracetam    |            |
|      0.0 +-------------------+-------------------+ P(BBB+)    |
|         0.0                 0.5                 1.0           |
|                                                              |
|   S_CNS = P(BBB+) x (1 - P(P-gp+))                           |
+---------------------------+----------------------------------+
                            v
+--------------------------------------------------------------+
| B9.15  display_full_scientific_report  -- 6 sections         |
|   1 quadrant | 2 BBB SHAP | 3 P-gp SHAP                      |
|   4 shared properties | 5 applicability domain | 6 CF        |
+---------------------------+----------------------------------+
                            v
+--------------------------------------------------------------+
| B9.16  optional exports  (raw) HTML / CSV / JSON   via C6    |
+--------------------------------------------------------------+
```

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **B9.1** | `argparse` with 7 flags; `os.system('color')` on Windows to enable ANSI | argv | namespace | none — I/O |
| **B9.2–B9.3** | `xgb.XGBClassifier(); .load_model(...)` for both barriers | ⬓ JSON | 2 boosters | none — I/O |
| **B9.4** | `load_scaler_metadata` for both; **BBB call passes hard-coded `default_bbb_means/stds`, P-gp call passes none** | ▱ JSON | 2 `(means,stds)` pairs | none — I/O |
| **B9.5** | Construct `DualBarrierShapExplainer`, `MorganSubstructureMapper`, `SharedPropertiesAnalyzer`, `ApplicabilityDomainEstimator`, `MolecularCounterfactualEngine` | models + scalers | 5 C-layer objects | none — construction |
| **B9.6** | 3-way branch: `--test-benchmark` → B9.17; `--smiles` → single-shot; else interactive REPL (`exit`/`quit`/`bench`) | flags | control | none — control |
| **B9.7–B9.8** | `extract_features_bbb` (B3.5) and `extract_features_pgp` (B3.6) on the same SMILES | SMILES | 2 feature tuples | none — deterministic |
| **B9.9** | `if bbb_feats is None or pgp_feats is None` → error; `sys.exit(1)` in non-interactive mode | tuples | control | none — control |
| **B9.10–B9.11** | `prepare_dataframe(..., is_pgp=False/True)` (B3.7) | features + scalers | `[1×2054]`, `[1×2060]` | none — deterministic |
| **B9.12–B9.13** | `predict_proba(df)[0][1]` | DataFrames | `P(BBB+)`, `P(P-gp+)` ∈ [0,1] | none — inference |
| **B9.14** | `pred = int(prob > 0.5)` — **hard-coded threshold, both barriers** | probabilities | 2 binary labels | none — threshold |
| **B8.1** | `determine_dual_barrier_quadrant(bbb_pred, pgp_pred)` → `(name, description)` | 2 ints | 2 strings | none — rule table |
| **B9.15** | 6-section terminal report (quadrant, BBB SHAP, P-gp SHAP, shared props, applicability domain, counterfactuals) | all above | stdout + `report_data` dict | none — presentation |
| **B9.16** | `ScientificReportGenerator.export_html/csv/json` (C6) | `report_data` | ▱ files | none — I/O |
| **B9.17** | `run_benchmark_test` — loops the 5 `BENCHMARK_DRUGS`, predicts, prints quadrant vs ground truth | — | stdout | none — regression check |

#### B8.1 · The four-quadrant truth table (draw as a 2×2 grid, it is the project's thesis)

| `bbb_pred` | `pgp_pred` | Quadrant | Label | Clinical exemplar |
|---|---|---|---|---|
| 1 | 0 | **Q1** | Optimal CNS Candidate | Levetiracetam, Caffeine, Diazepam |
| 1 | 1 | **Q2** | Paradoxical Drug / High Efflux Risk | **Loperamide** |
| 0 | 0 | **Q3** | Peripheral Low-Permeation Agent | Penicillin G |
| 0 | 1 | **Q4** | Restricted Effluxed Peripheral Compound | — |

**Derived scalar (used by D4, not by B8):**

```
  S_CNS  =  P(BBB+)  x  ( 1 - P(P-gp+) )           CNS desirability index,  in  [0,1]
```

> **Draw the quadrant block as a 2×2 grid with the axes being the two *independent* model outputs.**
> The scientific claim of the whole repository lives in this one picture: BBB permeability and P-gp
> efflux are **decoupled endpoints**, so Q2 exists as a real and populated category. A single-endpoint
> BBB model collapses Q1 and Q2 into one "permeable" bucket and therefore cannot distinguish
> Levetiracetam (works in brain) from Loperamide (never reaches brain).

#### `BENCHMARK_DRUGS` (hard-coded regression anchors in `decision_matrix.py`)

| Drug | Expected quadrant | Grounding |
|---|---|---|
| Levetiracetam (Keppra) | 1 | Evades P-gp entirely (Löscher & Potschka 2005) |
| Loperamide (Imodium) | 2 | Crosses membranes, actively extruded by luminal P-gp |
| Caffeine | 1 | High transcellular permeability, negligible P-gp recognition |
| Diazepam (Valium) | 1 | Lipophilic benzodiazepine, rapid brain distribution |
| Penicillin G | 3 | Ionised, high desolvation penalty, poor BBB penetration |

---

## SECTION 6 — EXPLAINABILITY AND VALIDATION LAYERS

### 6.1 Stage C1 · TreeSHAP margin decomposition

Source: `explainability/shap_explainer.py`

```text
+----------------------------------------------------------------+
| C1.0  compatibility patch                                      |
|   strip brackets from base_score:  '[5E-1]'  ->  '5E-1'        |
|   (XGBoost >= 2.0 writes a bracketed string)                   |
+-----------------------------+----------------------------------+
                              v
+----------------------------------------------------------------+
| C1.1  build shap.TreeExplainer x2, each in its own try/except  |
+-----------------------------+----------------------------------+
                              v
                     <> C1.2  route   BBB | PGP
                              v
+----------------------------------------------------------------+
| C1.3  predict_proba -> prob_pos, prob_neg                      |
|       predicted_class = int(prob_pos >= 0.5)                   |
+-----------------------------+----------------------------------+
                              v
                     <> explainer available ?
                     |
        +------------+-------------+
        v yes                      v no / raised
+----------------------+   +---------------------------------------+
| C1.4  TreeSHAP       |   | C1.5  FALLBACK  ** NOT SHAPLEY **     |
|   phi  = values[0]   |   |   x_j * (gain_j / sum gain) * logit(p)|
|   phi0 = base_values |   |   base_value = 0.0                    |
|   EXACT, ADDITIVE    |   |   breaks f(x) = phi0 + sum phi        |
|   polynomial time    |   |   violates local accuracy             |
+----------+-----------+   +------------------+--------------------+
           +------------------+---------------+
                              v
+================================================================+
|        f(x)   =   phi0   +   SUM phi_i                         |
|                                                                |
|        in LOG-ODDS MARGIN space, NOT probability space.         |
|        This is what makes BBB and P-gp attributions            |
|        directly comparable in C4.                              |
+=============================+==================================+
                              v
+----------------------------------------------------------------+
| C1.6  partition attributions                                   |
|   continuous_shap   = 6 (BBB) or 12 (P-gp) named descriptors   |
|   fp_active_shap    = ONLY bits whose input value == 1         |
+-----------------------------+----------------------------------+
                              v
                     <> C1.7  class-aligned ranking
                     |
                     +-- class 1 --> supporting = top +phi
                     |               opposing   = most -phi
                     +-- class 0 --> ROLES INVERT
                              |
                              v
+----------------------------------------------------------------+
| C1.8  enrich_feature_list                                      |
|   attach substructures for FP_* names  --> calls C2            |
+-----------------------------+----------------------------------+
                              v
+----------------------------------------------------------------+
| C1.9  biophysical rule engine                                  |
|   ** PRIOR KNOWLEDGE, NOT MODEL-DERIVED **                     |
|   hard-coded thresholds on RAW descriptors; printed BESIDE     |
|   the SHAP numbers and CAN DISAGREE with them                  |
+-----------------------------+----------------------------------+
                              v
  OUT:  base_value, model_margin, top_supporting, top_opposing,
        continuous_shap, interpretation,
        is_tree_shap   <-- computed but NEVER printed by B9.15
```

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **C1.0** | Monkey-patch `shap.explainers._tree.decode_ubjson_buffer` to strip `'[5E-1]'` → `'5E-1'` from `base_score` (XGBoost ≥ 2.0 writes a bracketed string) | — | patched function | none — compatibility |
| **C1.1** | `shap.TreeExplainer(model)` per barrier, each in its own `try/except`; sets `SHAP_AVAILABLE` from import success | boosters | 2 explainers or `None` | none — construction |
| **C1.2** | `is_pgp = model_name.upper()=="PGP"`; route model + explainer | name | model, explainer | none — control |
| **C1.3** | `predict_proba(input_df)[0]`; `predicted_class = int(prob_pos >= 0.5)` | `[1×d]` | `prob_pos`, `prob_neg`, class | none — inference |
| **C1.4** | `shap_res = explainer(input_df)`; `values[0]` → `φ ∈ ℝ^d`; `base_values[0]` → `φ₀` | `[1×d]` | `φ[d]`, `φ₀` | **exact Shapley values in log-odds margin space** |
| **C1.5** | **Fallback heuristic** (only if C1.4 raised): `contrib_j = x_j · (gain_j / Σ gain) · logit(p)` with `base_value = 0.0`, using `booster.get_score(importance_type='gain')` | `[1×d]` | pseudo-`φ[d]` | **not Shapley — see audit flag** |
| **C1.6** | Split into `continuous_shap` (6 or 12 named descriptors) and `fp_active_shap` (only bits where `input_df[c]==1`) | `φ` | 2 dicts | none — deterministic |
| **C1.7** | If `predicted_class==1`: supporting = top `+φ`, opposing = most negative `φ`. If `0`: **roles invert** | `φ` | 2 Series of length ≤ `top_n=5` | none — ranking |
| **C1.8** | `enrich_feature_list` — per feature emits `{feature, shap_value, input_value, is_continuous, is_active_bit, raw_value?, substructures?}`; calls C2 for `FP_*` names | Series | `list[dict]` | none — deterministic |
| **C1.9** | `_generate_biophysical_interpretation` — threshold rules on raw descriptors, keyed to the predicted class (see below) | raw descriptors | `{mechanism, biophysical_basis, rule_alignment}` | none — rule engine |

**The additive-attribution identity (this is what makes TreeSHAP exact, not approximate):**

```
                        d
                         d
  f(x)  =   phi_0   +   SUM   phi_i         in LOG-ODDS MARGIN space,
                        i=1                 NOT probability space
                      i=1

  implemented as:  raw_margin = base_value + shap_series.sum()
  returned as:     "output_space": "Log-odds raw margin (f(x) = base_value + sum(shap_values))"
```

> **Why margin space matters.** `φ` sums to the raw margin, not to the probability. Attributions
> are therefore additive and **comparable across the two models**, which is the precondition for the
> entire C4 cross-task comparison. If attributions were in probability space the sum would be
> non-additive through the sigmoid and C4's side-by-side `φ` table would be meaningless.
> Draw C1.4's output as a bar that sums to `f(x) − φ₀`.

#### C1.9 · Biophysical rule engine thresholds (fully enumerated)

| Model | Class | Fires when | Narrative asserted |
|---|---|---|---|
| BBB | 1 (permeant) | `LogP ≥ 1.5` | favourable membrane partition thermodynamics |
| BBB | 1 | `TPSA ≤ 90 Å²` | low desolvation penalty |
| BBB | 1 | `MolWt ≤ 400 Da` | CNS-penetrant diffusion rate |
| BBB | 1 | `HBD ≤ 2` | limits dipole water trapping |
| BBB | 0 (non-permeant) | `TPSA > 90` | excessive hydration-shell shedding cost |
| BBB | 0 | `LogP < 1.0` | insufficient lipophilicity |
| BBB | 0 | `MolWt > 450` | retarded transcellular diffusion |
| BBB | 0 | `HBD > 3` or `HBA > 7` | strong aqueous solvation clusters |
| P-gp | 1 (substrate) | `MolWt ≥ 400` or `Rg ≥ 4.0 Å` | spans the ~6,000 Å³ translocation cavity |
| P-gp | 1 | `HBD ≥ 2` or `HBA ≥ 4` | Seelig repeating donor/acceptor pharmacophore |
| P-gp | 1 | `RotBonds ≥ 6` | induced-fit binding in transmembrane helices |
| P-gp | 0 (evader) | `MolWt < 380` **and** `Rg < 3.8` | too compact to span multiple contact points |
| P-gp | 0 | `HBD ≤ 1` | avoids Seelig donor-pair trigger |
| P-gp | 0 | `RotBonds ≤ 5` | rigid scaffold resists entrapment |

Cited anchors: Pajouhesh & Lenz (2005) CNS guidelines, Seelig (1998) pharmacophore, Broccatelli
(2011) 3D cavity recognition, Löscher & Potschka (2005) efflux phenotypes.

> **Audit flag — C1.9 is a rule engine, not an explanation of the model.** These thresholds are
> hard-coded medicinal-chemistry heuristics evaluated on the raw descriptors. They are printed
> *beside* the SHAP attributions and can disagree with them. When drawing, put C1.9 in a visually
> distinct box labelled **"prior knowledge, not model-derived"** so it is never confused with C1.4.

> **Audit flag — C1.5 is not Shapley.** The fallback produces values that do not satisfy
> local accuracy, missingness, or consistency, and sets `base_value = 0.0`, so `f(x) = Σφ` no longer
> equals the model margin. The only honest signal that it fired is the returned
> `is_tree_shap: False` flag — which **the terminal report in B9.15 never prints**. Recommend
> surfacing it. Draw C1.5 with a dashed amber border.

---

### 6.2 Stage C2 · Morgan bit → substructure mapper

Source: `explainability/substructure_mapper.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **C2.1** | `get_bit_info` — `AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=d)` | `Mol` | `dict[bit_id → list[(atom_idx, radius)]]` | none — deterministic |
| **C2.2** | `extract_substructure_smiles(bit_id)`; returns `[]` if bit absent | `Mol`, `bit_id` | `list[dict]` | none — deterministic |
| **C2.3** | Radius-0 branch: fragment = atom symbol, lower-cased if aromatic | `Atom` | str | none — deterministic |
| **C2.4** | Radius>0 branch: `FindAtomEnvironmentOfRadiusN` → `PathToSubmol` → `MolToSmiles`; on exception emit `[Sym]*rN` | `Mol` | fragment SMILES | none — deterministic |
| **C2.5** | `_annotate_functional_group` — ordered substring cascade on the lower-cased fragment SMILES | str | functional-group tag | none — heuristic |

**C2.5 cascade order (first match wins — order is semantically significant):**

1. `c(=o)n` / `nc(=o)` → Amide Linkage / Peptidic Motif
2. `c(=o)o` → Ester / Carboxylate Group
3. `c=o` → Carbonyl / Ketone Functionality
4. `n(` or centre `N` → Basic/Aromatic N Heterocycle, else Aliphatic Amine
5. `c1ccccc1` / `c1ccncc1` / ≥5 lower-case `c` → Aromatic Hydrophobic Core
6. `s` → Sulfur-Containing Moiety
7. `cl`/`f`/`br` → Halogenated Substituent
8. `o` → Ether / Hydroxyl Oxygen
9. fallback → Carbon Backbone / Hydrocarbon Fragment

> **Audit flag.** Rule 6 tests `"s" in s` on a lower-cased string, so it also matches the `s` inside
> `"cs"`, and — more importantly — rule 7's halogen test runs *after* it, so a thioether-containing
> chlorinated fragment is tagged sulfur rather than halogenated. Also, rule 4's `center_symbol == "N"`
> check is case-sensitive against an already-lower-cased haystack. These are cosmetic labelling
> quirks, not numerical errors, but worth a footnote since the tags appear in the HTML report.

---

### 6.3 Stage C3 · Applicability domain

Source: `explainability/applicability_domain.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **C3.1** | `_load_reference_fingerprints` — reads `B3DB_classification.tsv` and `pgp_broccatelli.tab`, optional stride subsample via `max_samples`, builds Morgan FPs into `self.bbb_fps` / `self.pgp_fps` | ▱ raw datasets | 2 FP lists | none — I/O |
| **C3.2** | `BulkTanimotoSimilarity(test_fp, ref_fps)` → `max_tanimoto` | `Mol` | float ∈ [0,1] | none — deterministic |
| **C3.3** | Descriptor leverage: `norm_dist = sqrt(mean(scaled_desc²))` — **RMS of the z-scores** | `scaled_desc` | float ≥ 0 | none — deterministic |
| **C3.4** | 3-way domain gate (thresholds below) | `max_tanimoto`, `norm_dist` | status + warning | none — rule |
| **C3.5** | Uncertainty: `margin = abs(p − 0.5)`; `confidence = min(1, margin/0.5)`; `uncertainty = 1 − confidence` | `prob_pos` | 3 floats | none — deterministic |

**C3.4 gate, exactly as coded:**

```
  Inside Domain              :  max_tanimoto > 0.40  AND  norm_dist < 2.5
  Borderline Domain          :  max_tanimoto > 0.28  AND  norm_dist < 3.5
  Outside Applicability Domain :  otherwise
```

> **Note the docstring/code mismatch.** The comment block documents the borderline band as
> "Tanimoto 0.30 – 0.40", but the executed condition uses **0.28**. Code wins; the comment is stale.
> Also note the gate is evaluated as an if/elif chain, so a molecule with Tanimoto 0.45 but
> `norm_dist = 3.0` falls through to **Borderline**, not Outside — draw C3.4 as a 3-branch diamond
> cascade, not as a 2×2 grid, or you will mis-render that case.

> **Note the leverage metric is RMS of z-scores, not true Mahalanobis distance.** The module
> docstring says "Mahalanobis/Euclidean"; the implementation is `sqrt(mean(z²))`, which ignores the
> descriptor covariance matrix. For correlated descriptors (`MolWt` and `RotBonds` are strongly
> correlated) this under-penalises off-manifold molecules. Label the box
> **"RMS z-score, covariance-free"**.

---

### 6.4 Stage C4 · Cross-task shared properties

Source: `explainability/shared_properties.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **C4.1** | `compare_compound_shared_properties` — for each of the 6 shared descriptors, pull `φ_BBB` and `φ_Pgp` from the two C1 result dicts | 2 SHAP dicts | 6 table rows | none — deterministic |
| **C4.2** | `_interpret_shared_continuous_property` — per-descriptor sign logic (table below) | `prop`, raw value, 2 `φ` | `(relation, interpretation)` | none — rule engine |
| **C4.3** | `_find_shared_active_substructures` — set intersection of active-bit feature names appearing in **both** models' top-5 supporting+opposing lists | 2 SHAP dicts | `list[dict]` | none — deterministic |
| **C4.4** | `_synthesize_task_divergence` — scans relations for substrings `"Divergent"` / `"Synergistic"`, emits prose | table rows | str | none — deterministic |
| **C4.5** | `compute_global_shared_feature_correlation` (classmethod) — `spearmanr` over all shared feature importances, and separately over the 6 continuous ones if ≥3 present | 2 importance Series | `ρ`, `p` | **Spearman rank correlation** |

**C4.2 sign logic — the pharmacological core of the cross-task story:**

| Descriptor | `φ_BBB` | `φ_Pgp` | Relation emitted |
|---|---|---|---|
| `LogP` | > 0 | > 0 | **Antagonistic (Divergent)** — same property helps influx *and* efflux |
| `LogP` | > 0 | ≤ 0 | **Synergistic (Optimal)** — the CNS sweet spot |
| `LogP` | ≤ 0 | any | Concordant Impairment |
| `MolWt` | < 0 | > 0 | **Compound Barrier Burden** — blocks diffusion *and* feeds pump contacts |
| `MolWt` | ≥ 0 | ≤ 0 | Synergistic (Compact) |
| `MolWt` | other | other | Neutral / Moderate Impact |
| `TPSA` | < 0 | < 0 | Polarity Trade-off |
| `TPSA` | ≥ 0 | any | Permeation Favorable |
| `TPSA` | < 0 | ≥ 0 | Substrate Hindrance |
| `HDonors` / `HAcceptors` | any | > 0 | **Seelig Pharmacophore Trigger** |
| `HDonors` / `HAcceptors` | any | ≤ 0 | Restricted H-Bonding |
| `RotBonds` | any | > 0 | Induced-Fit Risk |
| `RotBonds` | any | ≤ 0 | Rigid Framework |

**C4.3 direction note matrix (shared active Morgan bits):**

| `φ_BBB` | `φ_Pgp` | Interpretation |
|---|---|---|
| > 0 | > 0 | Reinforcing efflux vulnerability |
| > 0 | < 0 | **Ideal CNS motif** |
| < 0 | > 0 | Double negative |
| else | else | Concordant barrier resistance |

---

### 6.5 Stage C5 · Counterfactual analog engine

Source: `explainability/counterfactual.py`

```text
  IN:  parent SMILES,  current_bbb_pred,  current_pgp_pred
            |
            v
   //======== C5.1  loop over 8 reaction SMARTS ========\\
            v                                           ||
+---------------------------------------+               ||
| C5.2  ReactionFromSmarts              |               ||
|       RunReactants + SanitizeMol      |               ||
|       skip on exception               |               ||
+------------------+--------------------+               ||
                   v                                    ||
        <> C5.3  already in seen_smiles ?               ||
        |                                               ||
        +-- yes --> skip ------------------------------=||
        +-- no                                          ||
                   |                                    ||
                   v                                    ||
        <> C5.4  Tanimoto >= 0.50 ?                     ||
        |        (chemical-sanity constraint)           ||
        +-- no  --> REJECT ---------------------------=||
        +-- yes                                         ||
                   |                                    ||
                   v                                    ||
+---------------------------------------+               ||
| C5.5  _evaluate_candidate             |               ||
|   re-extract Morgan + 2D + 3D         |               ||
|   ETKDGv3, maxAttempts = 50 on retry  |               ||
|   z-score with the SAME stored scalers|               ||
|   predict BOTH barriers               |               ||
|   returns None if embedding fails     |               ||
|   /!\ DUPLICATES core.features --     |               ||
|       highest-value refactor in repo  |               ||
+------------------+--------------------+               ||
                   v                                    ||
        <> C5.6  flip test                              ||
        |                                               ||
        +-- BBB 0 -> 1 ? --+                            ||
        +-- Pgp 1 -> 0 ? --+--> ACCEPT                  ||
        +-- neither -------+--> discard --------------=||
                   |                                    ||
                   v                                    ||
        <> C5.7  len >= 2 x max_counterfactuals ?        ||
        |                                               ||
        +-- no  --------------------------------------=//
        +-- yes --> break
                   |
                   v
+---------------------------------------+
| C5.8  sort by Tanimoto DESCENDING     |
|       return top max_counterfactuals  |
|       (default 3)                     |
+---------------------------------------+

This is a CONSTRAINED DISCRETE SEARCH, not an optimiser.
No gradient, no loss -- a boolean predicate plus a similarity ranking.
Transformations 1 and 6 are mutual inverses, as are 2 and 7.
```

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **C5.1** | Iterate the 8-entry `TRANSFORMATIONS` list of `(SMARTS, description)` | `Mol` | reaction objects | none — enumeration |
| **C5.2** | `ReactionFromSmarts` → `RunReactants((mol,))`; `Chem.SanitizeMol` then `MolToSmiles` inside `try/except` | `Mol` | candidate SMILES | none — deterministic |
| **C5.3** | Reject if already in `seen_smiles` (seeded with the canonical parent) or empty | SMILES | unique candidates | none — dedup |
| **C5.4** | `TanimotoSimilarity(orig_fp, cand_fp)`; **reject if `< min_similarity = 0.50`** | 2 FPs | float | **chemical-sanity constraint** |
| **C5.5** | `_evaluate_candidate` — recompute Morgan FP, 6 2D descriptors, ETKDGv3 embed (`maxAttempts=50` on retry), 6 3D descriptors; z-score with the **same** stored scalers; build `[1×2054]` and `[1×2060]`; predict both probabilities; `return None` if embedding fails | `Mol` | `{bbb_pred, bbb_prob, pgp_pred, pgp_prob, delta_descriptors}` | none — inference |
| **C5.6** | Flip test: success iff (`target ∈ {BBB,both}` **and** `0→1` on BBB) **or** (`target ∈ {Pgp,both}` **and** `1→0` on P-gp) | predictions | bool + note | **discrete objective: flip the class** |
| **C5.7** | `break` once `len(counterfactuals) >= max_counterfactuals * 2` | — | control | none — budget |
| **C5.8** | Sort by `tanimoto_similarity` descending; slice `[:max_counterfactuals]` (default 3) | list | `list[dict]` | **maximise similarity subject to flip** |

**The 8 transformations, exactly as coded:**

| # | SMARTS | Medicinal-chemistry intent |
|---|---|---|
| 1 | `[O;H1:1]>>[O:1]C` | Alcohol O-methylation: ↓TPSA, ↓HBD, ↑LogP |
| 2 | `[N;H2;!$(N*=O):1]>>[N:1]C` | Primary amine N-methylation: ↓HBD |
| 3 | `[C:1](=O)[O;H1:2]>>[C:1](=O)OC` | Carboxylate methyl esterification |
| 4 | `[C:1](=O)[O;H1:2]>>[C:1](=O)N` | Acid → primary amide bioisostere |
| 5 | `[c;H1:1]>>[c:1]F` | Aromatic fluorination: ↑LogP, ↑metabolic stability |
| 6 | `[O:1]C>>[O:1][H]` | Ether O-demethylation: ↓LogP |
| 7 | `[N;H0;!$(N*=O):1]C>>[N:1][H]` | Tertiary → secondary amine demethylation |
| 8 | `[C:1](=O)[C:2]>>[C:1][C:2]` | Ketone deoxygenation: ↓HBA |

> **This block is a constrained discrete search, not an optimiser.** Draw it as a loop with a
> rejection gate (Tanimoto ≥ 0.50) and an acceptance gate (class flip). There is no gradient and no
> loss — the "objective" is a boolean predicate plus a similarity ranking. Note that transformations
> 1 and 6 are mutual inverses, as are 2 and 7, so the enumeration can propose undoing its own edit
> on a different functional group.

> **Audit flag — C5.5 duplicates the feature kernel.** It re-implements descriptor extraction and
> z-scoring inline with `Descriptors`/`Descriptors3D` calls and hard-coded name lists rather than
> calling `core.features.extract_features_*` and `prepare_dataframe`. Any future change to B3 (a
> 7th descriptor, a different fingerprint radius) will silently desynchronise the counterfactual
> engine from the deployed models. Draw a dashed red line from B3 to C5.5 labelled
> **"duplicated, not shared"** — this is the highest-value refactor in the repository.

---

### 6.6 Stage C6 · Report generation

Source: `explainability/report_generator.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **C6.1** | `export_json` — `json.dump(report_data, indent=2)` after `makedirs` | dict | ▱ `.json` | none — I/O |
| **C6.2** | `export_csv` — flattens to one row: SMILES, timestamp, quadrant, both labels/probabilities, both domain statuses, both max-Tanimoto values, plus `Desc_*` columns | dict | ▱ single-row `.csv` | none — I/O |
| **C6.3** | `export_html` — 6-card standalone document with inline CSS, `render_feature_table` helper, green/red `φ` colouring, counterfactual cards, regulatory disclaimer | dict | ▱ `.html` | none — I/O |

Existing artifacts: `results/loperamide_report.csv`, `results/loperamide_report.html`.

---

### 6.7 Stage D1 · Scientific validation harness

Source: `scripts/run_validation.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **D1.1** | `calculate_metrics_with_ci` — point estimates for 10 metrics + TP/FP/TN/FN; `specificity = tn/(tn+fp)` computed manually | `y_true, y_pred, y_prob` | metric dict | none — measurement |
| **D1.2** | Bootstrap: `n_bootstraps=1000`, `RandomState(42)`, `rng.randint(0,n,n)` resampling **with replacement**; `continue` if a resample is single-class; `(2.5, 97.5)` percentiles | arrays | 95% CI per metric | none — resampling |
| **D1.3** | BBB in-distribution eval — re-derives the **exact** 15% stratified test split from B4.2b, applies the stored scaler metadata, loads the deployed booster, predicts | ⬓ CSV + ⬓ model + ▱ JSON | metrics + `bbb_test_confusion_matrix.png` | none — measurement |
| **D1.4** | `generate_bemis_murcko_split` — `MurckoScaffold.MurckoScaffoldSmiles` per molecule, cluster by scaffold, sort clusters by size descending, greedily fill test until `target_test_size` | DF | `(train_idx, test_idx)`, **0% scaffold overlap** | none — splitter |
| **D1.5** | Scaffold OOD eval — **trains a fresh untuned XGBoost** (`n_estimators=300, lr=0.05, max_depth=6` for BBB; `max_depth=5` for P-gp), fits a fresh `StandardScaler` on the scaffold-train partition | scaffold split | metrics | `binary:logistic` logloss |
| **D1.6** | P-gp in-distribution eval — re-derives the 20% stratified split from B5.3, applies stored scaler, deployed booster | ⬓ CSV + ⬓ model | metrics + `pgp_test_confusion_matrix.png` | none — measurement |
| **D1.7** | Export 10 metrics × 4 columns (BBB random/scaffold, P-gp random/scaffold) | all metrics | ⬓ `validation_metrics_summary.csv` | none — I/O |

**Measured results (from `results/validation_metrics_summary.csv`, estimate with 95% bootstrap CI):**

| Metric | BBB random | BBB scaffold | P-gp random | P-gp scaffold |
|---|---|---|---|---|
| Accuracy | 0.8856 [0.8659–0.9035] | 0.8598 [0.8393–0.8786] | 0.9295 [0.8963–0.9585] | 0.8838 [0.8423–0.9212] |
| Balanced accuracy | 0.8775 | **0.7605** | 0.9287 | 0.8978 |
| Precision | 0.9122 | 0.8849 | 0.9313 | 0.9829 |
| Recall / sensitivity | 0.9073 | 0.9415 | 0.9385 | 0.8156 |
| Specificity | 0.8478 | **0.5795** | 0.9189 | 0.9800 |
| F1 | 0.9097 | 0.9123 | 0.9349 | 0.8915 |
| ROC-AUC | 0.9522 | 0.9048 | 0.9587 | 0.9631 |
| PR-AUC | 0.9724 | 0.9668 | 0.9685 | 0.9801 |
| MCC | 0.7535 | **0.5718** | 0.8580 | 0.7843 |
| Brier score | 0.0847 | 0.0990 | 0.0679 | 0.0811 |

> **Read the scaffold column, not the random column.** The BBB model's specificity collapses from
> 0.848 to **0.580** and MCC from 0.754 to **0.572** under zero-scaffold-overlap evaluation, while F1
> *rises* to 0.912. That combination is the signature of a model defaulting to the majority
> (BBB+) class on unfamiliar chemotypes: recall climbs to 0.942 precisely because it says "permeant"
> more often. Annotate this on the drawing next to D1.5 — **it is the most important honest number
> in the repository**, and it is invisible if you only read accuracy or F1.

> **Audit flag — D1.4's greedy split is biased in an unusual direction.** Clusters are sorted
> **largest-first** and each is placed in test if it fits the remaining budget, else in train. So the
> *most populous* scaffolds (including the empty-scaffold cluster of acyclic molecules, typically the
> single largest group) land in **test**, and any cluster bigger than the whole budget can never enter
> test. This does guarantee zero scaffold overlap, but it is not the conventional "hold out rare
> chemotypes" protocol — it holds out common ones. Label the box **"greedy largest-first, not
> random scaffold sampling"**.

> **Audit flag — D1.5 evaluates a different model.** The scaffold numbers come from a freshly
> trained, untuned 300-tree booster, **not** the deployed Optuna-tuned 494-tree model. This is
> methodologically necessary (the deployed model saw scaffold-test molecules during its own
> training), but it means the scaffold and random columns are **not two evaluations of one model**.
> Draw D1.5 with its own small model box so the distinction is unmistakable.

---

### 6.8 Stage D2 · Feature ablation

Source: `scripts/run_ablation_study.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **D2.1** | `evaluate_feature_subset_cv` — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | `X, y, feature_cols` | 5 folds | none — splitter |
| **D2.2** | **Within-fold** `StandardScaler` fitted on fold-train continuous columns only (`cont_cols` ∩ `ALL_CONT_DESCRIPTORS`), applied to fold-val | fold data | scaled | none — deterministic |
| **D2.3** | Fixed `XGBClassifier(n_estimators=300, lr=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', tree_method='hist', random_state=42)`; **no `scale_pos_weight`**, no early stopping | fold train | booster | `binary:logistic` logloss |
| **D2.4** | Per-fold accuracy, balanced accuracy, F1, ROC-AUC, PR-AUC, MCC; aggregate `mean ± std` | predictions | metric dict | none — measurement |
| **D2.5** | 5 P-gp configurations × 3 BBB configurations → CSV + 2-panel horizontal bar chart | configs | ⬓ `ablation_study_results.csv`, ▱ `feature_ablation_comparison.png` | none — I/O |

**Measured results (5-fold CV mean ± std, from `results/ablation_study_results.csv`):**

| Task | Configuration | Features | ROC-AUC | F1 | MCC |
|---|---|---|---|---|---|
| P-gp | 2D descriptors only | 6 | 0.9185 ± 0.0107 | 0.8731 | 0.7229 |
| P-gp | **3D shape only** | 6 | **0.8425 ± 0.0243** | 0.8042 | 0.5649 |
| P-gp | 2D + 3D shape | 12 | 0.9192 ± 0.0131 | 0.8667 | 0.7098 |
| P-gp | **Morgan + 2D** | 2,054 | **0.9509 ± 0.0070** | 0.9026 | 0.7899 |
| P-gp | Full (Morgan + 2D + 3D) | 2,060 | 0.9481 ± 0.0076 | 0.9008 | 0.7870 |
| BBB | 2D descriptors only | 6 | 0.9174 ± 0.0081 | 0.8913 | 0.6831 |
| BBB | Morgan only | 2,048 | 0.9393 ± 0.0054 | 0.8936 | 0.6906 |
| BBB | **Full hybrid (Morgan + 2D)** | 2,054 | **0.9467 ± 0.0035** | 0.9041 | 0.7241 |

> **The ablation contradicts the deployed P-gp architecture — state this plainly on the drawing.**
> `Morgan + 2D` (2,054 features, ROC-AUC **0.9509**) **outperforms** the full `Morgan + 2D + 3D`
> configuration (2,060 features, ROC-AUC **0.9481**) that is actually deployed, and the gap
> (−0.0028) is well inside the fold standard deviations (±0.007). Meanwhile `3D shape only` is by
> far the weakest P-gp configuration (0.8425). The honest reading: **the 3D conformer descriptors
> are not earning their cost on this dataset** — they add a 1.1% molecule loss at the embedding gate
> (B2.3), the dominant per-molecule runtime cost, and no measurable ROC-AUC. They remain defensible
> for *interpretability* (`RadiusOfGyration` carries real cavity-spanning meaning in C1.9) but the
> performance claim is not supported. Draw a dashed red annotation from D2 back to B2.3.
>
> Note also that D2's models carry **no `scale_pos_weight`** while the deployed models do, so the
> ablation ranks architectures under a slightly different objective than the one that was shipped.

---

### 6.9 Stage D3 · Global explainability analysis

Source: `scripts/run_explainability_analysis.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **D3.1** | Load both processed datasets and both deployed boosters | ⬓ | DFs + models | none — I/O |
| **D3.2** | BBB split `test_size=0.2` **unstratified**; P-gp split `test_size=0.2` stratified | DFs | test partitions | none — splitter |
| **D3.3** | Subsample `min(500, n)` BBB and `min(250, n)` P-gp rows (`random_state=42`) | test sets | samples | none — sampling |
| **D3.4** | `explainer._bbb_explainer(X_sample).values` and `_pgp_explainer(...)` — direct access to the private attributes | samples | `φ [n×d]` each | exact TreeSHAP |
| **D3.5** | `mean_abs = mean(abs(φ), axis=0)`; `mean_signed = mean(φ, axis=0)`, both as Series indexed by feature name | `φ` | 4 Series | none — aggregation |
| **D3.6** | Per shared descriptor: assemble `{BBB_Mean_Abs_SHAP, Pgp_Mean_Abs_SHAP, BBB_Signed_SHAP, Pgp_Signed_SHAP, Task_Category, Scientific_Interpretation}` | Series | 6 rows | none — deterministic |
| **D3.7** | `spearmanr` across the 6 descriptors' mean-abs importances | 2 lists | `ρ`, `p` | Spearman rank correlation |
| **D3.8** | 4 figures: BBB top-15, P-gp top-15 (3D in red, shared 2D in blue, Morgan in grey), cross-task paired bars, hand-drawn 4-quadrant matrix with 5 annotated reference drugs | Series | ▱ 4 PNGs | none — I/O |

**Measured global attributions (`results/shared_properties_quantitative_comparison.csv`):**

| Property | BBB mean \|φ\| | P-gp mean \|φ\| | BBB signed φ | P-gp signed φ | Category |
|---|---|---|---|---|---|
| `MolWt` | 0.6955 | 0.6292 | −0.0412 | +0.0200 | Compound Barrier |
| `LogP` | 0.3957 | **1.4074** | +0.0545 | +0.1139 | **Antagonistic** |
| `TPSA` | **0.9875** | 0.1266 | +0.2473 | +0.0268 | Desolvation Filter |
| `HDonors` | 0.2548 | 0.0680 | +0.0421 | +0.0129 | Efflux Trigger (Seelig) |
| `HAcceptors` | 0.1118 | 0.0479 | +0.0321 | −0.0190 | Efflux Trigger (Seelig) |
| `RotBonds` | 0.1591 | 0.0671 | +0.0139 | −0.0049 | Conformational Flexibility |

> **The apparent scientific signal — but read the audit flag below before using it.** As tabulated,
> `TPSA` dominates BBB (0.988) and is nearly irrelevant to P-gp (0.127), while `LogP` dominates P-gp
> (1.407) and ranks third for BBB (0.396). That asymmetry would say the two barriers are governed by
> **different physics** — desolvation penalty versus hydrophobic cavity affinity — which is exactly
> the quantitative justification for modelling them as two decoupled endpoints.
>
> It is a mechanistically plausible result and it agrees with the medicinal-chemistry literature the
> project cites. **But these specific numbers were computed on unscaled inputs (§ audit flag below),
> so they cannot carry that argument yet.** Draw the two mirrored horizontal bar charts, and mark
> the panel **"pending re-run"** until D3 is fixed.

> **Audit flag — D3 computes SHAP on UNSCALED inputs.** `X_bbb_test` and `X_pgp_test` come straight
> from the processed CSVs, which store **raw** descriptors (see B1.6 / B2.6). No scaler metadata is
> applied anywhere in `run_explainability_analysis.py`. The boosters were trained on z-scored
> descriptors, so their internal split thresholds live in z-space — roughly the −3 to +3 band.
> Feeding raw values puts every continuous descriptor orders of magnitude outside that band: a
> molecule of average mass arrives as `MolWt = 383.86` where the model expects `≈ 0`, which is
> about **384 standard deviations** above the training centre. Consequently almost every split on a
> continuous descriptor resolves to the same branch for almost every molecule.
>
> The attributions do **not** come out as zero — `TPSA` still shows the largest BBB magnitude
> (0.9875). But that is the point: `TreeExplainer` measures each feature's contribution relative to
> the node-cover background distribution, so a feature pinned to one extreme branch for all inputs
> still receives a large, *nearly identical* attribution on every molecule. What the table is
> measuring is therefore **distance from the training background**, not per-molecule descriptor
> sensitivity, and the molecule-to-molecule variation that makes a SHAP summary informative has been
> suppressed. The six continuous-descriptor magnitudes should be regenerated before being quoted.
> Morgan-bit attributions are unaffected, since binary features are never scaled.
>
> Additionally, the code comment reads "Exact held-out split matching bbb_model_training.py" but
> uses `test_size=0.2` **unstratified**, whereas B4.2b uses `0.15` **stratified** — so the BBB rows
> analysed here are a different, partly-overlapping-with-train set.
>
> **Fix:** apply `load_scaler_metadata` + z-score before C1.4, and change the BBB split to
> `test_size=0.15, stratify=y`. Draw this box with a red border and the note **"re-run after fixing
> scaling"**.
>
> **Do not assume the TPSA-vs-LogP ranking will survive the fix.** It is independently plausible on
> biophysical grounds, but nothing in the current output constitutes evidence for it, and the
> direction of change after rescaling is genuinely unpredictable — correcting the inputs changes
> which tree paths are taken for every molecule. Treat the re-run as a real experiment with an
> unknown result, not as a formality. This is the one place in the repository where a headline
> interpretive claim currently rests on a computation that needs redoing.

---

### 6.10 Stage D4 · FDA pharmacopeia virtual screen

Source: `scripts/prepare_repurposing_library.py`, `scripts/run_repurposing_screen.py`

```text
   (raw) repurposing_drugs_20200324.txt  +  repurposing_samples_20200324.txt
            |
            v
+------------------------------------------------+
| D4.0a  urllib download (User-Agent: Mozilla)   |
| D4.0b  merge on pert_iname                     |
|        filter clinical_phase == 'Launched'     |
| D4.0c  RDKit canonicalise + dedup on SMILES    |
+----------------------+-------------------------+
                       v
      (db) fda_approved_repurposing_compounds.csv   [2,405 drugs]
                       |
                       v
+------------------------------------------------+
| D4.1  ProcessPoolExecutor                      |
|       workers = min(8, cpu_count())            |
|       results re-sorted by original idx        |
+----------------------+-------------------------+
                       v
+------------------------------------------------+
| D4.2  worker: Morgan + 2D x6 + ETKDGv3 3D x6   |
+----------------------+-------------------------+
                       v
              <> 3D embed ok ?
              |
              +-- no --> D4.2b  ZERO-FILL all 6 3D descriptors
              |           has_3d = False
              |           /!\ molecule is RETAINED, not dropped
              |               (training B2.3 DROPPED these)
              |           /!\ Rg   -> -3.90 sigma
              |           /!\ NPR2 -> -8.51 sigma
              |           /!\ has_3d never written to output CSV
              +-- yes
                       |
                       v
+------------------------------------------------+
| D4.3  pickle cache -- READ UNCONDITIONALLY     |
|       (raw) repurposing_features_cache.pkl     |
|       /!\ no invalidation on input change      |
+----------------------+-------------------------+
                       v
+------------------------------------------------+
| D4.4  bulk z-score  (mat - means) / stds       |
|       BBB [N x 6]        Pgp [N x 12]          |
+----------------------+-------------------------+
                       v
+------------------------------------------------+
| D4.5  TWO batched predict_proba calls TOTAL    |
|       X_bbb [N x 2054]   X_pgp [N x 2060]      |
+----------------------+-------------------------+
                       v
+------------------------------------------------+
| D4.6  quadrant_id in {1,2,3,4}                 |
|       S_CNS = P(BBB+) x (1 - P(P-gp+))         |
+----------------------+-------------------------+
                       v
+------------------------------------------------+
| D4.7  BulkTanimoto vs 1,500 reference FPs      |
|       'Inside' if BOTH >= 0.35 else 'Borderline'|
|       /!\ own gate -- differs from C3.4        |
|           which uses 0.40 / 0.28               |
+----------------------+-------------------------+
                       v
      +----------------+----------------+----------------+
      v                v                v                v
 (db) full        (db) top CNS     (db) efflux      (raw) 2 figures
      screen           repurposing      resistance        landscape +
      CSV              non-CNS & Q1     CNS & Q2          disease area
                       & tanimoto>=0.35  sorted by
                       sorted by S_CNS   prob_pgp
```

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **D4.0a** | `urllib` download of the two Broad Hub tables with a `Mozilla/5.0` User-Agent; skip if present | URLs | ▱ 2 `.txt` | none — I/O |
| **D4.0b** | `read_csv(sep='\t', comment='!')`; inner-merge samples↔drugs on `pert_iname`; filter `clinical_phase == 'Launched'` | 2 DFs | DF | none — deterministic |
| **D4.0c** | `MolFromSmiles` → `MolToSmiles` canonicalisation; `drop_duplicates(subset=['smiles'])` | DF | ⬓ **2,405 drugs** | none — deterministic |
| **D4.1** | `ProcessPoolExecutor(max_workers=min(8, cpu_count()))` over row tuples; `tqdm(as_completed(...))`; results sorted by original `idx` to restore order | task list | records | none — parallelism |
| **D4.2** | Worker: Morgan FP + 6 2D descriptors + `calculate_3d_descriptors(seed=42)` | row tuple | record dict | none — deterministic |
| **D4.2b** | **3D fallback**: on embedding failure, zero-fill all six 3D descriptors and set `has_3d=False` — **the molecule is retained, not dropped** | failed record | zero-filled record | none — imputation |
| **D4.3** | Pickle cache round-trip at `repurposing_features_cache.pkl`; cache is read unconditionally if the file exists | records | ▱ `.pkl` | none — I/O |
| **D4.4** | Bulk NumPy z-scoring: `(desc_mat - means) / np.where(stds != 0, stds, 1.0)`, separately for BBB (6) and P-gp (12) | `[N×6]`, `[N×12]` | scaled matrices | none — deterministic |
| **D4.5** | Build `X_bbb [N×2054]` and `X_pgp [N×2060]` DataFrames; **two** `predict_proba` calls total | matrices | 2 probability vectors | none — inference |
| **D4.6** | Threshold at 0.5 → `quad_id ∈ {1,2,3,4}`; `cns_desirability = prob_bbb * (1 - prob_pgp)` | probabilities | quadrant + score | **ranking objective `S_CNS`** |
| **D4.7** | `BulkTanimotoSimilarity` against `ApplicabilityDomainEstimator(max_samples=1500)` reference FPs; `domain_status = 'Inside' if both max-Tanimoto ≥ 0.35 else 'Borderline'` | FPs | 2 floats + status | none — gating |
| **D4.8** | Three CSVs: full screen; non-CNS ∩ Q1 ∩ `tanimoto_bbb ≥ 0.35` sorted by `S_CNS` (repurposing leads); CNS ∩ Q2 sorted by `prob_pgp` (efflux-resistance profiles). Plus landscape scatter and disease-area stacked bars | results | ⬓ 3 CSVs, ▱ 2 PNGs | none — I/O |

> **Note the D4.7 threshold is 0.35, not C3.4's 0.40/0.28.** The screen implements its own simplified
> two-state domain gate inline rather than calling `evaluate_compound`. Draw it as a separate small
> gate box so the inconsistency with C3.4 is visible.

> **Note D4.2b's imputation changes the semantics versus B2.3.** Training **dropped** molecules whose
> conformer failed; the screen **zero-fills** them and scores them anyway. A zero-filled 3D vector
> z-scores to roughly `−3.9σ` on `RadiusOfGyration` and `−8.5σ` on `NPR2` — far outside anything the
> booster saw in training. The `has_3d` flag is computed and cached but **never written to the output
> CSV and never used to filter**, so affected drugs appear in the lead tables indistinguishably.
> **Fix:** propagate `has_3d` into `screen_rows` and exclude or flag `has_3d == False` rows.
> Draw this as an amber dashed path.

> **Audit flag — dead/fragile line in D4.7.** `test_fp = domain_estimator.bbb_fps[0].__class__(2048)`
> is a no-op immediately overwritten on the next line, and it raises `IndexError` if `bbb_fps` is
> empty (which happens silently whenever `B3DB_classification.tsv` is missing, since C3.1 swallows
> that failure in a `try/except`). The following line also imports the private
> `core.features._MORGAN_GEN_R2_2048`. Both should be removed in favour of the public helper.

**Produced artifacts:** `fda_repurposing_dual_barrier_screen.csv`,
`top_cns_repurposing_candidates.csv`, `cns_drugs_efflux_resistance_profiles.csv`,
`repurposing_pharmacopeia_landscape.png`, `repurposing_disease_area_distribution.png`.

---

### 6.11 Stage D5 · Cryo-EM docking validation

Source: `scripts/run_cryoem_docking_validation.py`

```text
+-------------------------------------------------+
| D5.1  download vina 1.2.7 win exe               |
|       (if absent or < 100 kB)                   |
+-----------------------+-------------------------+
                        v
+-------------------------------------------------+
| D5.2  download PDB 6QEX                         |
|       3.5 Angstrom cryo-EM human ABCB1          |
+-----------------------+-------------------------+
                        v
+-------------------------------------------------+
| D5.3  extract chain A                           |
|       keep  ATOM  AND  line[21] == 'A'          |
|       /!\ drops HETATM, so the co-crystallised  |
|           Taxol ligand and all waters go away   |
+-----------------------+-------------------------+
                        v
+-------------------------------------------------+
| D5.4  mk_prepare_receptor  ->  PDBQT            |
|       box center = (173.33, 166.74, 161.48)     |
|       box size   = 24.0 Angstrom cube           |
|       (derived from the Taxol/TA1 site)         |
+-----------------------+-------------------------+
                        v
        //====== per compound, 22 defined ======\\
                        v                        ||
+-------------------------------------------------+
| D5.5  ligand preparation                        |
|       AddHs -> Embed(seed 42) -> MMFF94 optimise|
|       -> meeko -> (raw) ligand.pdbqt            |
|   (o) MMFF94 force-field energy minimisation    |
+-----------------------+-------------------------+
                        v                        ||
+-------------------------------------------------+
| D5.6  AutoDock Vina subprocess                  |
|       exhaustiveness 8  *  num_modes 5          |
|   (o) Vina empirical scoring function           |
+-----------------------+-------------------------+
                        v                        ||
+-------------------------------------------------+
| D5.7  parse mode-1 affinity                     |
|       + MODEL 1 coords, columns 30-54           |
+-----------------------+-------------------------+
                        v                        ||
+-------------------------------------------------+
| D5.8  Kd = exp(dG / RT)                         |
|       RT = 0.59248 kcal/mol at 298.15 K         |
+-----------------------+-------------------------+
                        v                        ||
+-------------------------------------------------+
| D5.9  contact residues within 3.8 Angstrom      |
|       brute-force norm per receptor atom        |
+-----------------------+-------------------------+
                        |                        ||
            \\==========+== loop ===============//
                        |
                        |   21 of 22 reach the CSV
                        |   (Topotecan silently lost, no warning)
                        v
+-------------------------------------------------+
| D5.10  statistics                               |
|    Welch t-test        Q2 vs Q1                 |
|    Mann-Whitney U      Q2 < Q1                  |
|    Spearman rho        P(P-gp+) vs dG           |
|    /!\ the P(P-gp+) values are HARD-CODED       |
|        literals -- D5 never calls B5, so this   |
|        correlation is NOT reproducible from     |
|        the pipeline                             |
+-----------------------+-------------------------+
                        v
+-------------------------------------------------+
| D5.11  (raw) 2 publication figures              |
|    /!\ significance bracket coordinates and     |
|        the "p < 0.001" string are HARD-CODED,   |
|        not driven by the computed p value       |
+-------------------------------------------------+

MEASURED SEPARATION   Q2 mean dG = -8.94    Q1 mean dG = -6.83
                      difference = 2.11 kcal/mol  ~  35-fold in Kd
```

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **D5.1** | Download `vina_1.2.7_win.exe` if absent or `< 100 kB` | URL | ▱ `bin/vina.exe` | none — I/O |
| **D5.2** | `urlretrieve("https://files.rcsb.org/download/6QEX.pdb")` | URL | ▱ `6QEX.pdb` | none — I/O |
| **D5.3** | Line filter: `line.startswith('ATOM') and line[21]=='A'` — **drops HETATM, so the co-crystallised Taxol ligand and all waters are removed** | PDB | ▱ `pgp_chain_a.pdb` | none — deterministic |
| **D5.4** | `mk_prepare_receptor.exe --read_pdb ... -p ... --forgive_extra_bonds -x --box_center --box_size` | PDB | ▱ `pgp_chain_a.pdbqt` | none — tool call |
| **D5.5** | `AddHs` → `EmbedMolecule(randomSeed=42)` (fallback `useRandomCoords=True`) → `MMFFOptimizeMolecule` → `meeko.MoleculePreparation` → `PDBQTWriterLegacy` | SMILES | ▱ ligand `.pdbqt` | **MMFF94 force field energy minimisation** |
| **D5.6** | Vina subprocess with the fixed grid box, `--exhaustiveness 8 --num_modes 5` | receptor + ligand | ▱ `*_docked.pdbqt` + stdout | **Vina empirical scoring function** |
| **D5.7** | Parse the stdout row whose first token is `'1'` → `ΔG`; parse `MODEL 1` ATOM/HETATM `x,y,z` from columns 30–54 | stdout + pdbqt | `ΔG`, `coords` | none — parsing |
| **D5.8** | `kd = exp(ΔG / RT)` with `RT = 0.59248 kcal/mol` (`R = 1.9872e-3`, `T = 298.15 K`); format as nM/µM/mM | `ΔG` | `K_d` | none — thermodynamic conversion |
| **D5.9** | Brute-force `norm(lig_arr - atom_coord)` per receptor atom; residue included if any ligand atom is within **3.8 Å**; sorted by residue number | coords | residue list | none — geometry |
| **D5.10** | `ttest_ind(q2, q1, equal_var=False)`; `mannwhitneyu(q2, q1, alternative='less')`; `spearmanr(prob_pgp_ml, ΔG)` | `ΔG` arrays | `t, p`; `U, p`; `ρ, p` | **null-hypothesis tests** |
| **D5.11** | Boxplot with significance bracket + ML-vs-docking scatter with `np.polyfit` trendline and 6 annotated drugs | DF | ▱ 2 PNGs | none — I/O |

**Grid box (derived from the co-crystallised Taxol/TA1 site in 6QEX):**

```
  center = ( 173.33 , 166.74 , 161.48 )        size = 24.0 A cube
```

**Named key contact residues sought in the transmembrane cavity:**
Phe303, Tyr307, Phe343, Gln725, Phe728, Phe983.

**Measured docking results (`results/cryoem_pgp_docking_validation.csv`, 21 of 22 compounds):**

| Quadrant | N | Mean ΔG (kcal/mol) | Representative |
|---|---|---|---|
| **Q2** (efflux vulnerable) | 10 | **−8.94** | Vilazodone −11.37, Lurasidone −10.33, Loperamide −9.08 |
| **Q1** (efflux evaders) | 8 | **−6.83** | Levetiracetam −5.24, Caffeine −5.55, Diazepam −6.21 |
| **Q3** (peripheral controls) | 3 | **−6.59** | Penicillin G −9.00, Aspirin −5.72, Metformin −5.05 |

Separation between Q2 and Q1 is **2.11 kcal/mol**, which at 298 K corresponds to roughly a
**35-fold** difference in dissociation constant.

> **The orthogonal-validation claim and its three honest caveats.** The headline is that a
> sequence-and-structure-free ML prediction (`P(P-gp+)`) correlates with an independent physics-based
> binding energy computed inside a real 3.5 Å cryo-EM structure. That is a genuinely strong form of
> external validation and worth drawing as a closing feedback arrow from D5 back to B5.
>
> But annotate three caveats directly on that arrow:
> 1. **`prob_pgp_ml` is hard-coded, not computed.** The 22 probabilities in `BENCHMARK_COMPOUNDS`
>    are literals pasted into the script, so D5 never calls B5. The Spearman ρ it reports is
>    therefore not reproducible from the pipeline — re-running `pgp_model_training.py` would not
>    update them. **Fix:** compute them live via `core.features` + the booster.
> 2. **Topotecan is silently missing.** 22 compounds are defined; **21** reach the CSV. Ligand
>    preparation or docking failed for Topotecan and the loop `continue`s without a warning, so
>    Q1's N is 8, not 9. The commit message claims 22 drugs.
> 3. **Penicillin G docks at −9.00 kcal/mol**, stronger than seven of the eight Q1 evaders (only
>    Betamethasone at −9.22 beats it) and comparable to the mean Q2 substrate, despite a Q3
>    designation and `P(P-gp+) = 0.034`. Betamethasone itself is the mirror-image outlier: a Q1
>    evader that docks more strongly than six of the ten Q2 substrates. This is the expected
>    failure mode of a rigid-receptor empirical scoring function: it scores *cavity occupancy*, not
>    *transport competence*. Binding affinity is necessary but not sufficient for efflux — a
>    molecule must also be released on the extracellular side. Draw it as a labelled outlier point;
>    presenting it honestly strengthens the work far more than omitting it.
>
> Also note the `plt.plot([0,0,1,1], [-10.2,-10.5,-10.5,-10.2])` significance bracket and the
> `p < 0.001` string in Figure 2's title are **hard-coded coordinates and text**, not driven by the
> computed `p_val`/`rho` passed into the function. If the data changes, the annotation will lie.

---

### 6.12 Stage D6 · Regression test suite

Source: `tests/test_explainability.py`

| Block | Operation | IN | OUT | LOSS / OBJECTIVE |
|---|---|---|---|---|
| **D6.1** | `models_and_scalers` fixture (graceful `pytest`-absent fallback via a bare decorator) — loads both models, both scalers, and all five C-layer objects | ⬓ artifacts | context dict | none — fixture |
| **D6.2** | `test_quadrant_1_optimal_cns` — Levetiracetam must yield `bbb_pred==1, pgp_pred==0` and `"Quadrant 1" in quadrant` | SMILES | assertions | **behavioural contract** |
| **D6.3** | `test_quadrant_2_paradoxical_drug` — Loperamide must yield `bbb_pred==1, pgp_pred==1` | SMILES | assertions | **behavioural contract** |
| **D6.4+** | Further cases covering the remaining quadrants, TreeSHAP attribution structure, feature ordering, substructure mapping, applicability domain, counterfactuals, and all three export formats | — | assertions | behavioural contracts |

> This is the only block in the repository that encodes **pharmacological ground truth as an
> executable constraint**. Draw it as a clamp on the B9 inference path: if a refactor breaks the
> Loperamide Q2 prediction, the science has regressed even if every metric still looks fine.

---

## SECTION 7 — CROSS-CUTTING AUDIT FINDINGS

Collected here so you can mark them on the drawing in one pass. Ordered by impact.

### 7.1 Legacy evaluation scripts feed unscaled inputs to scaled models

`BBB/bbb_model_evaluation.py` (B6†) and `Pgp/pgp_model_evaluation.py` (B7†) both:

- re-split with `test_size=0.2` (BBB: **unstratified**; P-gp: stratified) — neither matches the
  training split (B4.2b is 0.15 stratified; B5.3 is 0.20 stratified),
- load the deployed booster, and
- call `predict`/`predict_proba` on **raw, unscaled** `X` straight from the processed CSV.

Every continuous-descriptor split in those boosters is therefore evaluated on the wrong side of its
threshold, so the metrics and figures these scripts print are not valid estimates of deployed
performance. `run_validation.py` (D1) supersedes both and does apply the stored scaler metadata.

**Recommendation:** delete both scripts, or reduce them to thin wrappers over D1. Their outputs
`figures/pgp_confusion_matrix.png` and `figures/pgp_roc_curve.png` are still tracked in the
repository and should be regenerated from D1's `*_test_confusion_matrix.png` path before any
of those figures is used in the manuscript. Mark both boxes with a **†** and a red border.

### 7.2 Scaling-consistency matrix (the single most error-prone axis in this repo)

Draw this as a checklist beside the diagram:

| Block | Applies stored scaler? | Correct? |
|---|---|---|
| B4 training (BBB) | fits on train, transforms val/test | ✅ |
| B5 training (P-gp) | fits on train, transforms test | ✅ |
| B9 inference | `prepare_dataframe` with loaded metadata | ✅ |
| C5 counterfactual | inline z-score with the same stored means/stds | ✅ (but duplicated code — §6.5) |
| D1 validation | loads JSON metadata and applies it | ✅ |
| D2 ablation | fresh scaler fitted within each fold | ✅ |
| D4 screen | bulk NumPy z-score with loaded metadata | ✅ |
| **D3 global SHAP** | **no scaling applied** | ❌ §6.9 |
| **B6† BBB eval** | **no scaling applied** | ❌ §7.1 |
| **B7† P-gp eval** | **no scaling applied** | ❌ §7.1 |

Seven of ten call sites are correct. The three failures all share one root cause: the processed CSVs
store raw descriptors (by design, for leakage safety), so **any** new script that reads them must
apply `load_scaler_metadata` before calling a booster. That invariant is currently implicit.
**Recommendation:** add a `core.features.load_processed_dataset(path, is_pgp)` helper that returns
already-scaled features, and route every consumer through it.

### 7.3 Lower-severity findings

| # | Location | Finding |
|---|---|---|
| 1 | `train_gnn.py` A7.2 | `source == "Li"` filter silently discards part of the corpus; 1,890 graphs on disk, Adenot-only in training |
| 2 | `train_gnn.py` A7.15 | Only per-fold checkpoints written; `BEST_MODEL`/`FINAL_MODEL` constants unused, so `predict.py` falls back to fold 1 |
| 3 | `graph_generation.py` A4.2 | Atomic number and hybridisation enum passed as raw ordinals; no input normalisation anywhere in Paradigm A |
| 4 | `predict.py` A10.7 | `torch.mean(dim=0)` replaces `global_mean_pool`; correct only for single-molecule input |
| 5 | `predict.py` A10.2 | Inference model class is a re-declaration, not an import — latent drift from the training class |
| 6 | `bbb_data_preprocessing.py` | Imports `StandardScaler` and declares `SCALER_METADATA_PATH`; uses neither |
| 7 | `shap_explainer.py` C1.5 | Non-Shapley fallback; `is_tree_shap: False` computed but never surfaced in the terminal report |
| 8 | `applicability_domain.py` C3.3 | "Mahalanobis" in docstring, RMS z-score in code; docstring says borderline 0.30, code uses 0.28 |
| 9 | `counterfactual.py` C5.5 | Feature extraction duplicated instead of reusing `core.features` — desync risk |
| 10 | `run_repurposing_screen.py` D4.7 | Dead `__class__(2048)` line; `IndexError` if reference FPs empty; imports private `_MORGAN_GEN_R2_2048` |
| 11 | `run_repurposing_screen.py` D4.2b | `has_3d` computed and cached but never exported or used to filter |
| 12 | `run_repurposing_screen.py` D4.3 | Pickle cache read unconditionally; no invalidation if the input CSV or scalers change |
| 13 | `run_cryoem_docking_validation.py` D5 | `prob_pgp_ml` hard-coded; Topotecan silently missing; figure annotations hard-coded |
| 14 | `run_validation.py` D1.4 | Greedy largest-first scaffold allocation puts the most common scaffolds in test |
| 15 | `run_ablation_study.py` D2.3 | No `scale_pos_weight`, unlike the deployed models |
| 16 | `pgp_model_training.py` B5.5 | Test set present in `eval_set`; currently harmless, one keyword from leakage |
| 17 | `graph_generation.py` A4.3 | Edge feature index 4 redundant with index 3 |
| 18 | `substructure_mapper.py` C2.5 | Substring cascade ordering mis-tags sulfur-containing halogenated fragments |

---

## SECTION 8 — COMPLETE LOSS / OBJECTIVE INVENTORY

The whole repository contains exactly **eleven** optimisation or scoring objectives. Everything else
is a deterministic transform. Draw each as a hexagon.

| # | Block | Type | Exact form | Optimiser / engine |
|---|---|---|---|---|
| 1 | **A7.12** | Trainable loss | Weighted softmax cross-entropy, `w_c = N/(2n_c)`, reduction `mean` | AdamW + `clip_grad_norm_(2.0)` |
| 2 | **A7.10 / A7.15** | Selection metric | **Val F1** (drives LR schedule and checkpointing — *not* the loss) | `ReduceLROnPlateau(mode=max)`, early stop p=20 |
| 3 | **B4.4b** | Trainable loss | `binary:logistic` logloss + `Ω = γT + ½λ‖w‖² + α‖w‖₁` | XGBoost 2nd-order boosting, ES 30 |
| 4 | **B4.4** | Search objective | **Maximise F1 on `X_val`**, non-differentiable | Optuna TPE, 50 trials |
| 5 | **B4.5a** | Trainable loss | same as #3, with `best_params` | XGBoost, ES 50 → 494 trees |
| 6 | **B5.5** | Trainable loss | same functional form, `γ=0, α=0, λ=1` | XGBoost, no ES → 300 trees |
| 7 | **C1.4** | Attribution identity | `f(x) = φ₀ + Σφ_i` in log-odds margin space | TreeSHAP (exact, polynomial-time) |
| 8 | **C5.4 + C5.6/C5.8** | Constrained discrete search | maximise Tanimoto s.t. `Tanimoto ≥ 0.50` **and** class flip | exhaustive enumeration over 8 SMARTS |
| 9 | **D1.2** | Interval estimator | 1,000-resample non-parametric bootstrap, `(2.5, 97.5)` percentiles | `np.random.RandomState(42)` |
| 10 | **D4.6** | Ranking objective | `S_CNS = P(BBB+) · (1 − P(P-gp+))` | sort descending |
| 11 | **D5.5 / D5.6 / D5.10** | Physics + statistics | MMFF94 minimisation; Vina empirical `ΔG`; `K_d = exp(ΔG/RT)`; Welch *t*, Mann-Whitney *U*, Spearman *ρ* | RDKit / AutoDock Vina / SciPy |

> **The one-sentence summary to write under your diagram:** only four blocks in this repository
> actually *learn* (A7.12, B4.4b, B4.5a, B5.5); two blocks *search* without gradients (B4.4, C5);
> and five blocks *measure* (A7.10, C1.4, D1.2, D4.6, D5.10). Everything else is deterministic
> cheminformatics plumbing.

---

## SECTION 9 — SHAPE LEDGER (quick reference while drawing)

### Paradigm A

| Symbol | Meaning | Value |
|---|---|---|
| `V` | atoms in a molecule | variable, typically 10–60 |
| `E` | bonds | variable |
| `2E` | directed edges after A4.4 | `2 × E` |
| `d_v` | node feature width | **9** |
| `d_e` | edge feature width | **7** |
| `H` | hidden channels | **256** |
| `B` | batch size | **32** |
| logits | classifier output | `[B, 2]` |
| params | learnable | **345,514** |

### Paradigm B

| Symbol | Meaning | Value |
|---|---|---|
| FP width | Morgan bits, radius 2 | **2,048** |
| BBB continuous | 2D descriptors | **6** |
| P-gp continuous | 2D + 3D descriptors | **12** |
| BBB feature vector | model input width | **2,054** |
| P-gp feature vector | model input width | **2,060** |
| BBB trees | boosting rounds kept | **494** (ES at 443/2000) |
| P-gp trees | boosting rounds kept | **300** (full budget) |
| BBB train/val/test | stratified 70/15/15 | **5,463 / 1,171 / 1,171** |
| P-gp train/test | stratified 80/20 | **964 / 241** |
| BBB class balance | `BBB-` / `BBB+` | **2,849 / 4,956** (63.5% positive) |
| P-gp class balance | non-substrate / substrate | **555 / 650** (53.9% positive) |
| Screen cohort | FDA Launched | **2,405** |
| Docking cohort | defined / in results | **22 / 21** (Topotecan missing) |

### Paradigm A corpus

| Symbol | Meaning | Value |
|---|---|---|
| Curated corpus | `cleaned_bbb_dataset.csv` | **1,890** |
| Adenot subset | **actually trained on** | **1,621** |
| Li subset | discarded by A7.2 | **269** |
| CV folds | `StratifiedKFold` | **10** (≈146 test molecules per fold) |

---

## SECTION 10 — HAND-DRAWING PROTOCOL

A concrete order of operations for producing the full diagram on paper or a whiteboard.

### Canvas setup

Landscape, 4 horizontal bands. Suggested proportions on A3 or a 2×1 whiteboard:

```
 +-------------------------------------------------------------------------+
 | BAND 1 * INPUT             (height 8%)   SMILES string, single source   |
 +-------------------------------------------------------------------------+
 | BAND 2 * FEATURISATION     (height 26%)  A1-A5 left | B1-B3 right       |
 +-------------------------------------------------------------------------+
 | BAND 3 * MODELS + LOSSES   (height 38%)  A7-A8 left | B4-B5-B8 right    |
 +-------------------------------------------------------------------------+
 | BAND 4 * XAI + VALIDATION  (height 28%)  C1-C6 and D1-D6, right only    |
 +-------------------------------------------------------------------------+
```

Split bands 2 and 3 vertically down the middle: **Paradigm A on the left, Paradigm B on the right,
with no horizontal arrows crossing the divide.** Band 4 spans only the right half — Paradigm A has
no explainability or validation layer, and that visual asymmetry is itself a finding worth showing.

### Draw order

1. **Band 1.** One rounded box: `SMILES`. Two arrows down, one into each half.
2. **Band 2 left (A1–A5).** Vertical chain of 9 rectangles A2.1→A2.9. Hang the A3 image branch off
   A2.9 as a dead-end stub. Then A4.2 and A4.3 as two small boxes feeding A4.5. Beside A4.5, draw
   the three tensors as labelled rectangles: `x [V×9]`, `edge_index [2×2E]`, `edge_attr [2E×7]`.
   Write the 9 node features and 7 edge features as two numbered mini-tables in the margin.
3. **Band 2 right (B1–B3).** The B3 kernel as **one wide central box** with three internal
   compartments: `Morgan 2048`, `2D ×6`, `3D ×6 (ETKDGv3)`. Two arrows out: a 6-wide path to BBB and
   a 12-wide path to P-gp. Draw the B2.3 conformer gate as a diamond with a labelled reject arrow
   (`1,219 → 1,205`). Draw two **separate** scaler cylinders, clearly labelled with different means.
4. **Band 3 left (A8).** The seventeen A8 sub-blocks as a vertical stack. Annotate every arrow with
   its shape: `[V,9] → [V,256] → … → [B,256] → [B,32] → [B,2]`. Put the `global_mean_pool` box in a
   different colour — it is the only shape-changing operation that collapses the atom dimension.
   Place ⬡ A7.12 to the right of A8.17 with a dashed gradient arrow curving back up the whole stack.
   Write the loss formula in full in the margin.
5. **Band 3 right (B4, B5, B8).** B4 as **two nested boxes**: outer labelled "Optuna TPE, 50 trials,
   maximise val F1", inner labelled "XGBoost, logloss, ES 30". Draw ⬡ at the inner box with `g_i`
   and `h_i` arrows into a small "tree splitting" box. B5 as a single, visibly simpler box labelled
   "untuned, 300 trees, no ES". Draw `X_test` as a padlocked cylinder with exactly **one** outgoing
   arrow, to B4.6. Then the B8 2×2 quadrant grid with `P(BBB+)` on the x-axis and `P(P-gp+)` on the
   y-axis, the four quadrants labelled, and Levetiracetam / Loperamide / Penicillin G plotted.
6. **Band 4.** Five C boxes in a row feeding C6, then six D boxes below. Draw the D5→B5 feedback
   arrow last, in a distinct colour, labelled "orthogonal structural validation" — with the
   three caveats from §6.11 written beside it.
7. **Red-pen pass.** Mark the §7.2 scaling failures (D3, B6†, B7†), the D2-vs-B2.3 ablation
   contradiction (§6.8), the A7.2 `Li` filter, and the C5.5 duplicated kernel.

### The five things a reader must be able to see at a glance

If your finished drawing conveys only these, it has done its job:

1. **Two paradigms, no tensor exchange** — learned representation versus engineered representation.
2. **Two decoupled barriers** — which is why the 2×2 quadrant grid has four populated cells and why
   Loperamide and Levetiracetam separate at all.
3. **Four loss nodes, two search objectives, five measurement blocks** — and the fact that both
   paradigms are *trained* on a smooth surrogate (cross-entropy / logloss) but *selected* on a
   threshold metric (F1).
4. **The leakage firewall** — raw descriptors on disk, scaler fitted on train only, `X_test`
   touched exactly once. Plus the three places that firewall is currently breached.
5. **The scaffold-split honesty gap** — BBB specificity 0.848 → 0.580 on unseen chemotypes.

---

## SECTION 11 — ARTIFACT DEPENDENCY DAG

File-level build order. Anything downstream must be regenerated when an upstream artifact changes.

```text
PARADIGM A
  (raw) raw_adenot_li.xls
     |
     +--> (db) cleaned_bbb_dataset.csv
              |
              +--> (raw) graphs/*.pt
                       |
                       +--> (raw) best_bbb_gnn_model_fold_K.pt
                                |
                                +--> predict.py REPL     [runs on fold 1]


PARADIGM B -- BBB
  (raw) B3DB_classification.tsv
     |
     +--> (db) processed_b3db_data_advanced.csv
     |        |
     |        +--> (db)  bbb_xgboost_optuna_best.json ----+
     |        +--> (raw) bbb_scaler_metadata.json --------+
     |                                                    |
     +--> applicability-domain reference fingerprints ----+
                                                          |
PARADIGM B -- P-gp                                        |
  (raw) pgp_broccatelli.tab                               |
     |                                                    |
     +--> (db) processed_pgp_data.csv                     |
     |        |                                           |
     |        +--> (db)  pgp_xgboost_model.json ----------+
     |        +--> (raw) pgp_scaler_metadata.json --------+
     |                                                    |
     +--> applicability-domain reference fingerprints ----+
                                                          |
                                                          v
                                            +--------------------------+
                                            |      inference.py        |
                                            +--------------------------+


CONSUMERS OF THE TWO TRAINED MODELS
  bbb + pgp models  --> (db) validation_metrics_summary.csv                 (D1)
  processed CSVs    --> (db) ablation_study_results.csv                     (D2)
  bbb + pgp models  --> (db) shared_properties_quantitative_comparison.csv  (D3)


SCREEN
  (raw) repurposing_drugs + repurposing_samples
     |
     +--> (db) fda_approved_repurposing_compounds.csv
              |
              +--> (db) fda_repurposing_dual_barrier_screen.csv
                       |
                       +--> (db) top_cns_repurposing_candidates.csv
                       +--> (db) cns_drugs_efflux_resistance_profiles.csv

     (both models also feed the screen; the pickle cache sits in between)


DOCKING
  (raw) 6QEX.pdb --> (raw) pgp_chain_a.pdbqt --> (db) cryoem_pgp_docking_validation.csv

     ***  NO inbound edge from pgp_xgboost_model.json  ***


TWO STALENESS HAZARDS -- cases where the real dependency is not expressed in code

  1. The docking CSV does not depend on the P-gp model, because D5's ML
     probabilities are hard-coded literals. Retraining P-gp will NOT
     invalidate the reported correlation, even though it should.

  2. repurposing_features_cache.pkl (gitignored) sits between the FDA CSV
     and the screen output and is read unconditionally, so it will serve
     stale features after any change to B3 or to the input CSV.
```

> **Note the two staleness hazards visible in this DAG.** (1) `cryoem_pgp_docking_validation.csv` has
> **no inbound edge from `pgp_xgboost_model.json`**, because D5's ML probabilities are hard-coded
> (§6.11) — retraining P-gp will not invalidate the docking correlation, even though it should.
> (2) `repurposing_features_cache.pkl` (not shown, it is gitignored) sits between `FDA` and `SCR`
> and is read unconditionally, so it will serve stale features after any change to B3 or the input
> CSV. Both are cases where the real dependency is not expressed in code.

---

## SECTION 12 — PRIORITISED RECOMMENDATIONS

Ordered by scientific impact per unit of effort.

| # | Action | Why | Effort |
|---|---|---|---|
| 1 | Apply scaler metadata in `run_explainability_analysis.py` and fix the BBB split to `0.15` stratified; regenerate the shared-properties table and figures | The six continuous-descriptor SHAP values currently quoted are computed off-manifold (§6.9) | low |
| 2 | Compute `prob_pgp_ml` live in `run_cryoem_docking_validation.py`; log the Topotecan failure | Makes the headline orthogonal-validation correlation reproducible (§6.11) | low |
| 3 | RESOLVED 2026-10-24: both scripts and their two tracked figures were deleted. Originally: delete or rewrite `bbb_model_evaluation.py` and `pgp_model_evaluation.py`; regenerate their tracked figures from D1 | They report invalid metrics on unscaled inputs (§7.1) | low |
| 4 | Export `has_3d` in the screen output and exclude or flag `has_3d == False` rows | Zero-filled 3D vectors are scored ~4–8σ off-manifold (§6.10) | low |
| 5 | Add `core.features.load_processed_dataset(path, is_pgp)` returning scaled features; route all consumers through it | Makes the scaling invariant explicit instead of implicit (§7.2) | medium |
| 6 | Refactor `counterfactual._evaluate_candidate` to call `extract_features_*` + `prepare_dataframe` | Removes the only duplicated copy of the feature kernel (§6.5) | medium |
| 7 | Surface `is_tree_shap` in the terminal and HTML reports; consider removing the C1.5 fallback | A silent non-Shapley fallback is worse than a loud failure (§6.1) | low |
| 8 | Either drop the 3D descriptors from the deployed P-gp model or justify them on interpretability grounds in the manuscript | The ablation shows they cost 1.1% of molecules and buy no ROC-AUC (§6.8) | medium |
| 9 | Add `scale_pos_weight` to the ablation models | So the ablation ranks architectures under the shipped objective (§6.8) | low |
| 10 | Tune the P-gp model with Optuna, as the BBB model already is | Removes the largest remaining asymmetry between the two barriers (§4.2) | medium |
| 11 | Normalise Paradigm A node features, or one-hot atomic number and hybridisation | Raw ordinals 1–53 dominate boolean flags into the first linear layer (§2.2) | medium |
| 12 | Report Paradigm A metrics as Adenot-only, or remove the `Li` filter | Current numbers silently describe a subset of the stated corpus (§2.4) | low |

---

## APPENDIX A — EXECUTION ORDER (reproducing everything from scratch)

```bash
# -- Paradigm A --------------------------------------------------------------
cd "Initial Model"
python clean_raw_data.py          # A1-A3  > cleaned_bbb_dataset.csv + images/
python graph_generation.py        # A4-A5  > graphs/*.pt + graph_summary.csv
python gnn_dataset.py             # A6     (optional sanity check)
python check_graph.py             # A6.4   (optional single-graph inspection)
python train_gnn.py               # A7-A9  > best_bbb_gnn_model_fold_1..10.pt
python predict.py                 # A10    interactive REPL

# -- Paradigm B: preprocessing and training ---------------------------------
cd "../B3DB Based Model"
python BBB/bbb_data_preprocessing.py   # B1  > processed_b3db_data_advanced.csv
python Pgp/pgp_data_preprocessing.py   # B2  > processed_pgp_data.csv
python BBB/bbb_model_training.py       # B4  > bbb_xgboost_optuna_best.json + scaler
python Pgp/pgp_model_training.py       # B5  > pgp_xgboost_model.json + scaler

# -- Paradigm B: inference --------------------------------------------------
python inference.py --test-benchmark
python inference.py --smiles "CC[C@@H](C(=O)N)N1CCCC1=O" --explain --counterfactual
python inference.py --smiles "<SMILES>" --export-html results/report.html

# -- Validation and analysis ------------------------------------------------
python scripts/run_validation.py                  # D1 > validation_metrics_summary.csv
python scripts/run_ablation_study.py              # D2 > ablation_study_results.csv
python scripts/run_explainability_analysis.py     # D3 > global SHAP (FIX SCALING FIRST, sec 6.9)
python scripts/prepare_repurposing_library.py     # D4.0 > fda_approved_repurposing_compounds.csv
python scripts/run_repurposing_screen.py          # D4 > 3 CSVs + 2 figures
python scripts/run_cryoem_docking_validation.py   # D5 > docking CSV + 2 figures
python -m pytest tests/test_explainability.py -v  # D6 > 4-quadrant regression suite
```

---

## APPENDIX B — EXTERNAL DEPENDENCIES BY BLOCK

| Library | Minimum | Blocks that require it |
|---|---|---|
| `numpy` | 1.24 | everywhere |
| `pandas` | 2.0 | everywhere |
| `openpyxl` | 3.1 | A1 (`.xls` reading) |
| `rdkit` | 2023.9.1 | A2.8, A4, A10.4, B3 (all), C2, C3, C5, D4.0c, D5.5 |
| `scikit-learn` | 1.3 | A7.4, A7.14, B4.2, B4.3a, B5.3, D1, D2 |
| `xgboost` | 2.0 | B4, B5, B9, C1, D1, D2, D3, D4 |
| `optuna` | 3.4 | B4.4 |
| `shap` | 0.44 | C1 (with the C1.0 compatibility patch) |
| `torch` | 2.1 | A4.5, A5.1, A6, A7, A8, A10 |
| `torch_geometric` | 2.4 | A6 (`Data`, `Dataset`), A7.7 (`DataLoader`), A8 (`GINEConv`, `global_mean_pool`) |
| `matplotlib`, `seaborn` | 3.7 / 0.12 | B6†, B7†, D1, D2, D3, D4.8, D5.11 |
| `tqdm` | 4.66 | B2.2, D4.1 |
| `scipy` | (transitive) | C4.5, D3.7, D5.10 |
| `Pillow` | (transitive via rdkit) | A3.2 |
| `meeko` | **not pinned in `requirements.txt`** | D5.5 |
| AutoDock Vina 1.2.7 | downloaded at runtime by D5.1 | D5.6 |

> **Two packaging gaps.** `meeko` and `scipy` are imported but absent from `requirements.txt`, and
> `mk_prepare_receptor.exe` is resolved from `os.path.dirname(sys.executable)`, which only works
> inside the active virtual environment on Windows. Add `meeko` and `scipy` to `requirements.txt`
> and note the Windows-only constraint of D5 in the README.

---

*End of specification. Every block listed here was read directly from source at commit `a53789f`;*
*every numeric value was measured from the files on disk or extracted from the saved model JSONs,*
*not copied from prose documentation.*
