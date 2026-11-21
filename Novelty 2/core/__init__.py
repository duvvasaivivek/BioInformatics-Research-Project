"""
Core package for Novelty 2: Sequential Cascaded Sieve and Transfer Learning Architecture.
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
    load_scaler_metadata,
)

from .sequential_engine import (
    SequentialCascadedEngine,
    SequentialResult,
    BENCHMARK_DRUGS,
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
    "SequentialCascadedEngine",
    "SequentialResult",
    "BENCHMARK_DRUGS",
]
