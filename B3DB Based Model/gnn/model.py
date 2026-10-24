"""
Multi-Task Graph Neural Network: Shared Encoder, One Head Per Barrier.

The encoder is the GINEConv trunk from the base-paper reference implementation,
kept deliberately unchanged so that any difference in results is attributable to
multi-task training and feature encoding rather than to a new architecture:

    3 x [ GINEConv(edge_dim) -> BatchNorm1d -> ReLU -> Dropout ] -> global_mean_pool

The heads branch from the 256-dim POOLED graph vector, not from the 32-unit fc1
bottleneck the reference implementation uses. Branching after fc1 would force both
tasks through 32 shared units; branching at the pooled vector gives each task its
own 256 -> 32 -> 2 readout while still sharing all message passing.

Measured parameter cost (hidden=256, 3 layers):
    legacy 9-feature input,  1 head : 345,514  (matches the reference implementation)
    encoded 66-feature input, 1 head: 364,608
    encoded 66-feature input, 2 heads: 372,898
So the second barrier costs +8,290 parameters, about 2.3% of the model. Branching
after fc1 instead would cost only +66, but at the price of a 32-unit shared
bottleneck for both tasks.
"""

from typing import Dict, List, Optional, Any

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


def _mlp(in_dim: int, hidden: int) -> torch.nn.Sequential:
    """The 2-layer MLP each GINEConv wraps, as in the reference implementation."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
    )


class MultiTaskGNN(torch.nn.Module):
    """
    Shared GINEConv encoder with an independent classification head per task.

    Parameters
    ----------
    num_atom_features, num_bond_features
        Derived from the featurizer, never hardcoded. The reference implementation
        baked 9 and 7 in as literals in both the model and the checkpoint metadata,
        which is how metadata starts silently lying after a feature change.
    tasks
        Ordered task names, e.g. ('bbb', 'pgp'). One head is built per task.
    hidden_channels, dropout, num_layers
        Trunk geometry. Defaults match the reference implementation.
    residual
        Off by default to stay faithful to the base architecture. When on, each
        message-passing block adds a skip connection.
    """

    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int,
        tasks: List[str],
        hidden_channels: int = 256,
        dropout: float = 0.30,
        num_layers: int = 3,
        num_classes: int = 2,
        readout_dim: int = 32,
        residual: bool = False,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError('num_layers must be at least 1')
        if not tasks:
            raise ValueError('at least one task is required')

        self.tasks = list(tasks)
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_atom_features = num_atom_features
        self.num_bond_features = num_bond_features
        self.residual = residual

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        in_dim = num_atom_features
        for _ in range(num_layers):
            self.convs.append(GINEConv(_mlp(in_dim, hidden_channels), edge_dim=num_bond_features))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))
            in_dim = hidden_channels

        # One readout head per task, branching from the pooled 256-dim vector.
        self.heads = torch.nn.ModuleDict({
            task: torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, readout_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(readout_dim, num_classes),
            )
            for task in self.tasks
        })

    def encode(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        """Runs the shared trunk and returns the pooled graph embedding [B, hidden]."""
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Residual only once dimensions match (i.e. after the first layer).
            x = h + x if (self.residual and h.shape == x.shape) else h
        return global_mean_pool(x, batch)

    def forward(self, data, tasks: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Returns {task_name: logits [B, num_classes]}.

        A dict rather than a single tensor, because the reference implementation's
        single `output.argmax(dim=1)` contract cannot express two tasks.
        """
        graph_embedding = self.encode(data.x, data.edge_index, data.edge_attr, data.batch)
        selected = self.tasks if tasks is None else tasks
        return {task: self.heads[task](graph_embedding) for task in selected}

    def embed(self, data) -> torch.Tensor:
        """Pooled graph embedding only, for representation analysis."""
        return self.encode(data.x, data.edge_index, data.edge_attr, data.batch)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_metadata(self) -> Dict[str, Any]:
        """
        Self-describing metadata for the checkpoint.

        Everything here is read off the live module, so it cannot contradict the
        weights it is saved beside.
        """
        return {
            'tasks': list(self.tasks),
            'num_atom_features': self.num_atom_features,
            'num_bond_features': self.num_bond_features,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'residual': self.residual,
            'trainable_parameters': self.count_parameters(),
            'head_parameters': {
                task: sum(p.numel() for p in head.parameters() if p.requires_grad)
                for task, head in self.heads.items()
            },
        }


def build_model_from_checkpoint(checkpoint: Dict[str, Any]) -> MultiTaskGNN:
    """
    Reconstructs a model from a checkpoint's own metadata.

    Use this instead of re-declaring the architecture at the call site. The
    reference implementation's predict.py re-declares its model class with
    different dropout and a hand-rolled pooling function, and hardcodes a hidden
    size that disagrees with its own default checkpoint - so it raises a
    load_state_dict size mismatch as committed.
    """
    meta = checkpoint['architecture']
    model = MultiTaskGNN(
        num_atom_features=meta['num_atom_features'],
        num_bond_features=meta['num_bond_features'],
        tasks=meta['tasks'],
        hidden_channels=meta['hidden_channels'],
        dropout=meta['dropout'],
        num_layers=meta['num_layers'],
        num_classes=meta['num_classes'],
        residual=meta.get('residual', False),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
