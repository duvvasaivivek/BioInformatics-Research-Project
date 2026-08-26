# BBB Dataset — Zhao et al. (2007)

## Source File And This Explanation is for only the DataSet

**Supporting Information:** `ci600312d_si_001.xls`

This XLS is the Supporting Information associated with Zhao et al. (2007), *Predicting Penetration Across the Blood-Brain Barrier From Simple Descriptors and Fragmentation Schemes*.

It contains two main tables:

- **Table S1:** Adenot-derived data
- **Table S2:** Li-derived data

The tables have different columns because they originate from different source datasets.

---

## Table S1 — Adenot Dataset

Table S1 is the larger and more detailed table. It contains compound information, molecular structures, BBB classification, molecular descriptors, and training/test information.

| Attribute | Meaning |
|---|---|
| `No` | Compound identification number |
| `SMILES-Adenot` | Original SMILES from the Adenot dataset |
| `SMILES` | Processed/standardized SMILES |
| `SMILES-B` | Another SMILES representation |
| `Name` | Compound name |
| `BBB-crossing` | BBB classification |
| `P-gp` | Information related to P-glycoprotein |
| `MW` | Molecular weight |
| `PSA` | Polar surface area |
| `logP` | Lipophilicity-related descriptor |
| `logD(7.4)` | Distribution coefficient at approximately pH 7.4 |
| `No of H donors` | Number of hydrogen-bond donor sites |
| `No of H Acceptors` | Number of hydrogen-bond acceptor sites |
| `pKa` | Acid/base ionization property |
| `No of Rotatable bonds` | Approximate molecular flexibility |
| `Training or test set` | Original training/test assignment |

### Important terms

**SMILES:** A text representation of a chemical structure. Software such as RDKit can use SMILES to generate molecular fingerprints, descriptors, and molecular graphs.

**BBB-crossing:** The target classification indicating whether a compound is considered to penetrate the blood-brain barrier.

**BBB+:** BBB-penetrating/high-penetration compound.

**BBB−:** BBB-nonpenetrating/low-penetration compound.

**P-gp:** P-glycoprotein, a transporter protein that can affect whether compounds cross biological barriers.

**MW:** Molecular weight.

**PSA:** Polar surface area; a measure related to the polar portion of a molecule and relevant to permeability.

**logP:** Lipophilicity of a molecule.

**logD(7.4):** Lipophilicity/distribution at approximately physiological pH 7.4, considering ionization.

**H-bond donors/acceptors:** Molecular sites that can donate or accept hydrogen bonds.

**pKa:** A measure related to a molecule gaining or losing a proton and therefore its ionization/charge state.

**Rotatable bonds:** A rough measure of molecular flexibility.

### Dataset size

The Adenot-derived portion used by Zhao contains:

- **1,593 compounds**
- **1,283 BBB+**
- **310 BBB−**

It also contains training/test information used in the Zhao experimental setup.

---

## Table S2 — Li Dataset

Table S2 is smaller and has a simpler structure because it comes from the Li dataset.

Important columns include:

| Attribute | Meaning |
|---|---|
| `No` | Compound number in the supporting table |
| `No-Li` | Original compound ID from the Li dataset |
| `Name` | Compound name |
| `SMILES-Li` | SMILES from the Li dataset |
| `SMILES` | Processed/standardized SMILES |
| `Class` | BBB classification |
| `Ref.` | Reference/source information |

The Li-derived portion used in the Zhao dataset contributes:

- **397 compounds**
- **267 BBB+**
- **130 BBB−**

`Class` serves the same general purpose as `BBB-crossing` in Table S1: it identifies the BBB classification.

---

## Why S1 and S2 Are Different

They are **not two versions of the same table**.

```text
Adenot dataset
      ↓
Table S1
      ↓
Detailed molecular descriptors
Training/test information

Li dataset
      ↓
Table S2
      ↓
Compound + SMILES + BBB class + reference
```

Therefore, the different column names and attributes are expected.

---

## The 1,990-Compound Dataset

The two portions combine as:

```text
Adenot-derived data
1,593 compounds

+

Li-derived data
397 compounds

=

1,990 compounds
```

Class totals:

| Source | BBB+ | BBB− | Total |
|---|---:|---:|---:|
| Adenot-derived | 1,283 | 310 | 1,593 |
| Li-derived | 267 | 130 | 397 |
| **Total** | **1,550** | **440** | **1,990** |

So:

**1,550 BBB+ + 440 BBB− = 1,990 compounds**

---

## Summary

**Table S1 — Adenot-derived**

- 1,593 compounds
- 1,283 BBB+
- 310 BBB−
- Detailed molecular descriptors
- Training/test information

**Table S2 — Li-derived**

- 397 compounds
- 267 BBB+
- 130 BBB−
- Simpler compound/SMILES/class/reference information

Together:

**1,550 BBB+ + 440 BBB− = 1,990 compounds**

This XLS is therefore an important raw source for understanding and reconstructing the BBB datasets used by later research.
