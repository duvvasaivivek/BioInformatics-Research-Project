"""
Multi-Task Graph Neural Network Package for Dual-Barrier CNS Prediction.

Jointly learns blood-brain barrier permeability (head A) and P-glycoprotein efflux
substrate status (head B) from a single shared GINEConv molecular-graph encoder.

Scientific motivation: on the molecules where the B3DB and Broccatelli benchmarks
overlap, the two labels are near-independent (Cramer's V = 0.064, OR 1.42
[0.76, 2.69]). Near-independent labels carry complementary rather than redundant
information, which is the precondition for multi-task learning to help the smaller
task. Every published model learns these two endpoints separately.

Modules
-------
device        : CUDA device selection and GPU tuning. All device choices route here.
featurizer    : the single canonical SMILES -> torch_geometric Data path.
data_assembly : canonicalisation, InChIKey join keys, corpus merge, per-task masks.
splits        : molecule-level random and Bemis-Murcko scaffold partitioning.
model         : MultiTaskGNN, shared trunk with one head per task.
losses        : masked multi-task cross-entropy.
metrics       : classification metric suite with bootstrap confidence intervals.

Note: `Initial Model/` is the reference implementation of the base paper (GCN-BBB)
and is deliberately left untouched. This package is independent of it.
"""

from .device import get_device, describe_device, tune_backend, cuda_memory_summary
from .featurizer import (
    FEATURE_MODES,
    feature_dims,
    smiles_to_data,
    LEGACY_ATOM_FEATURE_NAMES,
    LEGACY_BOND_FEATURE_NAMES,
)

__all__ = [
    "get_device",
    "describe_device",
    "tune_backend",
    "cuda_memory_summary",
    "FEATURE_MODES",
    "feature_dims",
    "smiles_to_data",
    "LEGACY_ATOM_FEATURE_NAMES",
    "LEGACY_BOND_FEATURE_NAMES",
]
