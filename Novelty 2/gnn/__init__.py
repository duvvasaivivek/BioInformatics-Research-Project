"""
GNN package for Novelty 2: Sequential Transfer Learning Architecture.
"""

from .sequential_transfer_gnn import (
    SequentialTransferGNN,
    GINEEncoderTrunk,
    load_pretrained_bbb_encoder,
)

__all__ = [
    "SequentialTransferGNN",
    "GINEEncoderTrunk",
    "load_pretrained_bbb_encoder",
]
