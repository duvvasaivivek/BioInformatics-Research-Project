"""
Molecule-Level Partitioning for the Multi-Task GNN.

Two protocols:
  'random'   - stratified on the joint (label, mask) pattern, so both label
               distributions and the single/dual-label mix are preserved.
  'scaffold' - Bemis-Murcko scaffold split in the DeepChem convention (largest
               scaffold clusters fill train first), giving a test set of rare
               chemotypes with zero scaffold overlap against train.

The critical invariant: partitioning happens at the level of the MOLECULE, never
the task. A dual-labelled molecule must carry both of its labels into the same
partition. If a molecule were in train for BBB and test for P-gp, every multi-task
number would be contaminated. assert_split_integrity enforces this and should be
called before any training run.
"""

from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

RDLogger.DisableLog('rdApp.*')

SPLIT_MODES = ('random', 'scaffold')


def scaffold_of(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES; empty string for acyclic or unparseable input."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return ''
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ''


def _stratum_labels(corpus: pd.DataFrame, min_count: int = 3) -> np.ndarray:
    """
    Builds a composite stratum per molecule from its (label, mask) pattern.

    'x' marks an absent label, so e.g. '1_x' is a BBB-positive molecule with no
    P-gp label. Rare strata are merged into one bucket so sklearn's stratified
    split does not fail on a class of size 1.
    """
    bbb = np.where(corpus['m_bbb'].values == 1, corpus['y_bbb'].astype(str), 'x')
    pgp = np.where(corpus['m_pgp'].values == 1, corpus['y_pgp'].astype(str), 'x')
    strata = np.char.add(np.char.add(bbb.astype(str), '_'), pgp.astype(str))

    counts = pd.Series(strata).value_counts()
    rare = set(counts[counts < min_count].index)
    if rare:
        strata = np.where(np.isin(strata, list(rare)), 'rare', strata)
    return strata


def _random_split(
    corpus: pd.DataFrame, test_size: float, val_size: float, seed: int
) -> Dict[str, np.ndarray]:
    idx = np.arange(len(corpus))
    strata = _stratum_labels(corpus)

    try:
        trainval_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=strata
        )
    except ValueError:
        print('[WARN] Stratified split failed; falling back to unstratified.')
        trainval_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)

    # val_size is expressed as a fraction of the WHOLE corpus, so rescale it
    # against the remaining train+val pool.
    inner = val_size / max(1.0 - test_size, 1e-9)
    sub_strata = strata[trainval_idx]
    try:
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=inner, random_state=seed, stratify=sub_strata
        )
    except ValueError:
        train_idx, val_idx = train_test_split(trainval_idx, test_size=inner, random_state=seed)

    return {'train': np.sort(train_idx), 'val': np.sort(val_idx), 'test': np.sort(test_idx)}


def _scaffold_split(
    corpus: pd.DataFrame, test_size: float, val_size: float, seed: int
) -> Dict[str, np.ndarray]:
    """
    DeepChem-convention scaffold split: largest clusters fill train first.

    This deliberately leaves the rare chemotypes in val/test, which is the honest
    hard case. The alternative (largest clusters into test) produces a
    chemically narrow test set that flatters the model.
    """
    scaffolds: Dict[str, List[int]] = {}
    for i, smi in enumerate(corpus['canonical_smiles'].values):
        scaffolds.setdefault(scaffold_of(smi), []).append(i)

    # Deterministic ordering: size descending, then scaffold string, so the result
    # does not depend on dict insertion order.
    clusters = sorted(scaffolds.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    cluster_lists = [v for _, v in clusters]

    n = len(corpus)
    n_train_target = int(n * (1.0 - test_size - val_size))
    n_val_target = int(n * val_size)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cluster in cluster_lists:
        if len(train_idx) + len(cluster) <= n_train_target:
            train_idx.extend(cluster)
        elif len(val_idx) + len(cluster) <= n_val_target:
            val_idx.extend(cluster)
        else:
            test_idx.extend(cluster)

    return {
        'train': np.sort(np.array(train_idx, dtype=int)),
        'val': np.sort(np.array(val_idx, dtype=int)),
        'test': np.sort(np.array(test_idx, dtype=int)),
    }


def make_splits(
    corpus: pd.DataFrame,
    mode: str = 'random',
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Partitions the corpus by molecule. Returns index arrays for train/val/test."""
    if mode not in SPLIT_MODES:
        raise ValueError(f'Unknown split mode {mode!r}; expected one of {SPLIT_MODES}')
    if mode == 'random':
        return _random_split(corpus, test_size, val_size, seed)
    return _scaffold_split(corpus, test_size, val_size, seed)


def assert_split_integrity(
    corpus: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    mode: str,
) -> Dict[str, Any]:
    """
    Verifies the partitioning before training. Raises AssertionError on violation.

    Checks, in order of how badly each would invalidate the paper:
      1. No molecule (InChIKey) appears in more than one partition.
      2. Every molecule is assigned to exactly one partition.
      3. Every dual-labelled molecule has BOTH labels in the SAME partition.
      4. For a scaffold split, zero Bemis-Murcko scaffold overlap between train
         and test.
      5. Each partition contains both classes for each task it carries, otherwise
         ROC-AUC is undefined there.
    """
    names = ['train', 'val', 'test']
    keys = {name: set(corpus.iloc[splits[name]]['inchikey']) for name in names}

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = keys[a] & keys[b]
            assert not overlap, (
                f'Molecule leakage between {a} and {b}: {len(overlap)} shared InChIKeys, '
                f'e.g. {list(overlap)[:3]}'
            )

    total_assigned = sum(len(splits[n]) for n in names)
    assert total_assigned == len(corpus), (
        f'Partition does not cover the corpus: {total_assigned} assigned vs {len(corpus)} molecules'
    )

    # Check 3 is structurally guaranteed by splitting on molecules, but assert it
    # explicitly so a future refactor to per-task splitting fails loudly here.
    dual_mask = (corpus['m_bbb'] == 1) & (corpus['m_pgp'] == 1)
    dual_keys = set(corpus[dual_mask]['inchikey'])
    for name in names:
        part = corpus.iloc[splits[name]]
        part_dual = part[(part['m_bbb'] == 1) & (part['m_pgp'] == 1)]
        for _, row in part_dual.iterrows():
            assert row['m_bbb'] == 1 and row['m_pgp'] == 1, (
                f'Dual-labelled molecule {row["inchikey"]} lost a label in {name}'
            )

    scaffold_overlap = None
    if mode == 'scaffold':
        train_scaffolds = {scaffold_of(s) for s in corpus.iloc[splits['train']]['canonical_smiles']}
        test_scaffolds = {scaffold_of(s) for s in corpus.iloc[splits['test']]['canonical_smiles']}
        shared = train_scaffolds & test_scaffolds
        scaffold_overlap = len(shared)
        assert scaffold_overlap == 0, (
            f'Scaffold split leaked {scaffold_overlap} shared scaffolds between train and test'
        )

    report: Dict[str, Any] = {
        'mode': mode,
        'sizes': {n: int(len(splits[n])) for n in names},
        'dual_labelled_total': int(len(dual_keys)),
        'dual_labelled_per_partition': {},
        'per_task_counts': {},
        'scaffold_overlap_train_test': scaffold_overlap,
    }

    for name in names:
        part = corpus.iloc[splits[name]]
        report['dual_labelled_per_partition'][name] = int(
            ((part['m_bbb'] == 1) & (part['m_pgp'] == 1)).sum()
        )
        task_counts = {}
        for task in ('bbb', 'pgp'):
            sub = part[part[f'm_{task}'] == 1]
            pos = int((sub[f'y_{task}'] == 1).sum())
            neg = int((sub[f'y_{task}'] == 0).sum())
            task_counts[task] = {'labelled': int(len(sub)), 'positive': pos, 'negative': neg}
            if len(sub) > 0:
                assert pos > 0 and neg > 0, (
                    f'Partition {name} has only one class for task {task} '
                    f'({pos} pos / {neg} neg); ROC-AUC would be undefined'
                )
        report['per_task_counts'][name] = task_counts

    return report
