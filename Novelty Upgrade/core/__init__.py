"""
Core domain package for Dual-Barrier CNS Drug Predictor.
Exports feature extractors, scalers, and pharmacological decision logic.
"""

from .features import (
    BBB_DESCRIPTOR_NAMES,
    PGP_DESCRIPTOR_NAMES,
    calculate_2d_descriptors,
    calculate_3d_descriptors,
    calculate_morgan_fingerprint,
    extract_features_bbb,
    extract_features_pgp,
    prepare_dataframe,
    load_scaler_metadata
)
from .decision_matrix import (
    determine_dual_barrier_quadrant,
    BENCHMARK_DRUGS,
    Colors
)

__all__ = [
    "BBB_DESCRIPTOR_NAMES",
    "PGP_DESCRIPTOR_NAMES",
    "calculate_2d_descriptors",
    "calculate_3d_descriptors",
    "calculate_morgan_fingerprint",
    "extract_features_bbb",
    "extract_features_pgp",
    "prepare_dataframe",
    "load_scaler_metadata",
    "determine_dual_barrier_quadrant",
    "BENCHMARK_DRUGS",
    "Colors"
]
