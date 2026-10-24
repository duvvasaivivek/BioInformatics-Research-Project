"""
Graph Dataset Construction and Caching for the Multi-Task GNN.

Turns the merged dual-barrier corpus into a list of torch_geometric Data objects,
each carrying both task labels and both task masks as graph-level attributes.

Graph-level attributes are stored with shape [1] so PyG's collate concatenates
them to [B] across a batch, which is what the masked loss expects.

The whole set is cached as one torch file rather than ~8,800 individual .pt files.
Featurisation of the full corpus takes well under a minute, but caching keeps
repeated experiment runs fast.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .featurizer import smiles_to_data, feature_dims, describe_featurization

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'graph_cache')


# Bumped whenever the stored graph schema changes, so a stale cache is rebuilt
# rather than silently reused. v2 renamed corpus_index -> corpus_row (see below).
CACHE_VERSION = 2


def cache_path_for(mode: str) -> str:
    return os.path.join(CACHE_DIR, f'dual_barrier_graphs_{mode}_v{CACHE_VERSION}.pt')


def build_graph_dataset(
    corpus: pd.DataFrame,
    mode: str = 'encoded',
    use_cache: bool = True,
    verbose: bool = True,
) -> Tuple[List[Data], Dict[str, Any]]:
    """
    Featurises every molecule in the corpus.

    Returns (graphs, stats). Each graph carries:
        y_bbb, m_bbb, y_pgp, m_pgp : long/float tensors of shape [1]
        corpus_row                 : row position back into the corpus DataFrame
        inchikey, smiles           : identifiers for traceability

    Molecules RDKit cannot featurise are skipped and counted. The returned
    corpus_row lets every graph be traced back to its corpus row, so splits
    computed on the corpus can be applied to the graph list without assuming
    the two are the same length.

    The attribute is deliberately NOT called corpus_index: PyG's default
    Data.__inc__ increments any attribute whose key contains 'index' by the
    graph's node count when collating a batch, which silently corrupts a
    row pointer into out-of-range values.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = cache_path_for(mode)

    if use_cache and os.path.exists(cache):
        payload = torch.load(cache, weights_only=False)
        if payload.get('corpus_size') == len(corpus) and payload.get('version') == CACHE_VERSION:
            if verbose:
                print(f'[GRAPH] Loaded {len(payload["graphs"])} cached graphs '
                      f'({mode} features) from {os.path.relpath(cache, BASE_DIR)}')
            return payload['graphs'], payload['stats']
        if verbose:
            print('[GRAPH] Cache corpus size mismatch; rebuilding.')

    atom_dim, bond_dim = feature_dims(mode)
    graphs: List[Data] = []
    failed: List[str] = []

    for i, row in corpus.iterrows():
        data = smiles_to_data(
            row['canonical_smiles'],
            mode=mode,
            y_bbb=torch.tensor([int(row['y_bbb'])], dtype=torch.long),
            m_bbb=torch.tensor([float(row['m_bbb'])], dtype=torch.float32),
            y_pgp=torch.tensor([int(row['y_pgp'])], dtype=torch.long),
            m_pgp=torch.tensor([float(row['m_pgp'])], dtype=torch.float32),
            corpus_row=torch.tensor([int(i)], dtype=torch.long),
        )
        if data is None:
            failed.append(str(row['inchikey']))
            continue
        data.inchikey = str(row['inchikey'])
        data.smiles = str(row['canonical_smiles'])
        graphs.append(data)

    stats: Dict[str, Any] = {
        'featurization': describe_featurization(mode),
        'graphs_built': len(graphs),
        'featurization_failures': len(failed),
        'failed_inchikeys_sample': failed[:10],
        'num_atom_features': atom_dim,
        'num_bond_features': bond_dim,
        'mean_nodes': round(float(np.mean([g.num_nodes for g in graphs])), 2) if graphs else 0.0,
        'mean_edges': round(float(np.mean([g.edge_index.shape[1] for g in graphs])), 2) if graphs else 0.0,
    }

    if verbose:
        print(f'[GRAPH] Built {len(graphs)} graphs ({mode}: {atom_dim} atom / {bond_dim} bond features); '
              f'{len(failed)} failures')
        print(f'[GRAPH] Mean nodes {stats["mean_nodes"]}, mean directed edges {stats["mean_edges"]}')

    if use_cache:
        torch.save(
            {'graphs': graphs, 'stats': stats, 'corpus_size': len(corpus),
             'mode': mode, 'version': CACHE_VERSION},
            cache,
        )
        if verbose:
            print(f'[GRAPH] Cached to {os.path.relpath(cache, BASE_DIR)}')

    return graphs, stats


def subset_by_corpus_index(graphs: List[Data], indices: np.ndarray) -> List[Data]:
    """
    Selects the graphs whose corpus_row is in `indices`.

    Resolving through corpus_row rather than list position means a featurisation
    failure cannot silently shift the mapping between splits and graphs.
    """
    wanted = set(int(i) for i in indices)
    return [g for g in graphs if int(g.corpus_row.item()) in wanted]


def make_loader(
    graphs: List[Data],
    batch_size: int = 32,
    shuffle: bool = False,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Builds a DataLoader, with pinned memory when a CUDA device is in play.

    num_workers defaults to 0: the graphs are already in RAM, so worker processes
    add Windows spawn overhead without speeding anything up.
    """
    pin = bool(device is not None and device.type == 'cuda')
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin,
        num_workers=num_workers,
    )


def label_vector(graphs: List[Data], task: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (labels, mask) arrays for a task over a graph list, for diagnostics."""
    y = np.array([int(getattr(g, f'y_{task}').item()) for g in graphs])
    m = np.array([float(getattr(g, f'm_{task}').item()) for g in graphs])
    return y, m
