"""
Sequential Transfer Learning GNN Architecture for Novelty 2.

Solves the Multi-Task Negative Transfer problem identified in Novelty 1.
Instead of forcing simultaneous hard parameter sharing with competing gradient updates
over an uncorrelated joint label set (Cramér's V = 0.064), this module implements
Hierarchical Sequential Transfer:

1. Base Pre-training: GINEConv encoder pre-trained on full BBB dataset (B3DB, N=7,805).
   Learns rich general representations of molecular permeation, hydrogen bonding, and lipophilic topology.
2. Sequential Transfer: Pre-trained encoder weights (convs and norms) are transferred to
   initialize the downstream P-glycoprotein (P-gp) efflux predictor (N=1,205).
3. Adaptation: The transferred trunk can be either frozen (feature extractor) or fine-tuned
   with a low learning rate, while training a task-specific P-gp classification head.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


def _mlp(in_dim: int, hidden: int) -> torch.nn.Sequential:
    """The 2-layer MLP wrapped by each GINEConv message-passing block."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
    )


class GINEEncoderTrunk(torch.nn.Module):
    """
    Message-passing encoder trunk that processes molecular graphs into a pooled 256-dim embedding.
    """

    def __init__(
        self,
        num_atom_features: int = 66,
        num_bond_features: int = 14,
        hidden_channels: int = 256,
        dropout: float = 0.30,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_atom_features = num_atom_features
        self.num_bond_features = num_bond_features
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        in_dim = num_atom_features
        for _ in range(num_layers):
            self.convs.append(GINEConv(_mlp(in_dim, hidden_channels), edge_dim=num_bond_features))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
            in_dim = hidden_channels

    def forward(self, batch) -> torch.Tensor:
        """Computes pooled graph embedding: [batch_size, hidden_channels]."""
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return global_mean_pool(x, batch.batch)


class SequentialTransferGNN(torch.nn.Module):
    """
    Downstream P-gp Model with pre-trained BBB encoder trunk and dedicated classification head.
    """

    def __init__(
        self,
        trunk: GINEEncoderTrunk,
        readout_dim: int = 32,
        num_classes: int = 2,
        freeze_trunk: bool = False,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.freeze_trunk = freeze_trunk
        self.readout_dim = readout_dim
        self.num_classes = num_classes

        if freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad = False

        self.head = torch.nn.Sequential(
            torch.nn.Linear(trunk.hidden_channels, readout_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(trunk.dropout),
            torch.nn.Linear(readout_dim, num_classes)
        )

    def forward(self, batch) -> torch.Tensor:
        """Returns [batch_size, num_classes] logits."""
        embedding = self.trunk(batch)
        return self.head(embedding)

    def unfreeze_trunk(self) -> None:
        """Unfreezes trunk parameters for end-to-end fine-tuning."""
        self.freeze_trunk = False
        for p in self.trunk.parameters():
            p.requires_grad = True


def load_pretrained_bbb_encoder(checkpoint_path: str, device: torch.device) -> GINEEncoderTrunk:
    """
    Extracts the GINEConv encoder trunk weights from a pre-trained BBB model checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"BBB checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("architecture", {})
    num_atom = arch.get("num_atom_features", 66)
    num_bond = arch.get("num_bond_features", 14)
    hidden = arch.get("hidden_channels", 256)
    layers = arch.get("num_layers", 3)
    dropout = arch.get("dropout", 0.30)

    trunk = GINEEncoderTrunk(
        num_atom_features=num_atom,
        num_bond_features=num_bond,
        hidden_channels=hidden,
        dropout=dropout,
        num_layers=layers,
    ).to(device)

    # Filter state dict for convs and norms
    state_dict = ckpt["model_state_dict"]
    trunk_state = {}
    for k, v in state_dict.items():
        if k.startswith("convs.") or k.startswith("norms."):
            trunk_state[k] = v

    missing, unexpected = trunk.load_state_dict(trunk_state, strict=True)
    return trunk
