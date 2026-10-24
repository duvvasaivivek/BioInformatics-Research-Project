"""
K-Fold Out-of-Fold Prediction Runner for the Multi-Task GNN.

The single-split runs leave only 33 (random) or 48 (scaffold) dual-labelled
molecules in the test partition - far too few to stratify or to score quadrant
decisions on. K-fold over the whole corpus produces an honest out-of-fold
prediction for every labelled molecule, including all ~216 that carry both labels.

Folds are stratified on the (label, mask) composite from gnn/splits.py, so each
fold preserves both label distributions and the single/dual-label mix. A validation
slice is carved from each fold's training portion for early stopping, so the
held-out fold is never seen during training or model selection.

Example
-------
    python scripts/run_kfold_multitask.py --tasks bbb,pgp --folds 5 --tag multitask
    python scripts/run_kfold_multitask.py --tasks bbb      --folds 5 --tag single_bbb
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from gnn.device import (
    get_device, describe_device, tune_backend, cuda_memory_summary,
    reset_cuda_peak, empty_cuda_cache,
)
from gnn.data_assembly import build_corpus, label_independence
from gnn.dataset import build_graph_dataset, subset_by_corpus_index
from gnn.splits import _stratum_labels
from gnn.training import TrainConfig, train_one_model, predict_graphs, parse_task_weights
from gnn.metrics import classification_metrics, summarise_folds

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
ALL_TASKS = ('bbb', 'pgp')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='K-fold OOF predictions for the multi-task GNN.')
    p.add_argument('--tasks', type=str, default='bbb,pgp')
    p.add_argument('--task-weights', type=str, default='')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--val-fraction', type=float, default=0.15,
                   help='Validation slice carved from each fold training portion')
    p.add_argument('--features', type=str, default='encoded', choices=('encoded', 'legacy'))
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.30)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bootstraps', type=int, default=1000)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--tag', type=str, default='')
    return p.parse_args()


def assert_oof_integrity(
    corpus: pd.DataFrame,
    oof: pd.DataFrame,
    fold_train_rows: Dict[int, set],
    tasks: List[str],
) -> Dict[str, Any]:
    """
    The check that invalidates every downstream metric if it fails.

      1. No molecule receives more than one out-of-fold prediction.
      2. Every molecule that carries a label for an active task is covered.
      3. No molecule is scored by a fold that trained on it.
    """
    assert oof['corpus_row'].is_unique, (
        f'{len(oof) - oof["corpus_row"].nunique()} molecules received more than one '
        f'out-of-fold prediction'
    )

    for task in tasks:
        labelled = set(corpus.index[corpus[f'm_{task}'] == 1])
        scored = set(oof.loc[oof[f'{task}_pred'].notna(), 'corpus_row'])
        missing = labelled - scored
        assert not missing, (
            f'{len(missing)} molecules with a {task} label received no out-of-fold '
            f'prediction, e.g. {list(missing)[:3]}'
        )

    leaks = []
    for fold, train_rows in fold_train_rows.items():
        scored_rows = set(oof.loc[oof['fold'] == fold, 'corpus_row'])
        overlap = scored_rows & train_rows
        if overlap:
            leaks.append((fold, len(overlap)))
    assert not leaks, f'Molecules scored by a fold that trained on them: {leaks}'

    dual = (corpus['m_bbb'] == 1) & (corpus['m_pgp'] == 1)
    dual_rows = set(corpus.index[dual])
    dual_covered = set(oof.loc[
        oof['bbb_pred'].notna() & oof['pgp_pred'].notna(), 'corpus_row'
    ]) & dual_rows

    return {
        'molecules_scored': int(len(oof)),
        'unique': bool(oof['corpus_row'].is_unique),
        'dual_labelled_total': int(dual.sum()),
        'dual_labelled_with_both_predictions': int(len(dual_covered)),
        'train_test_leaks': 0,
    }


def main() -> int:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    for t in tasks:
        if t not in ALL_TASKS:
            raise ValueError(f'Unknown task {t!r}')
    weights = parse_task_weights(args.task_weights, tasks)
    active = [t for t in tasks if weights[t] > 0]
    tag = args.tag or ('multitask' if len(active) > 1 else f'single_{active[0]}')
    run_id = f'{tag}_kfold{args.folds}_{args.features}'
    started = time.time()

    device = get_device()
    backend = tune_backend(device)
    reset_cuda_peak(device)

    print('=' * 78)
    print(f'K-FOLD OUT-OF-FOLD PREDICTIONS: {run_id}')
    print('=' * 78)
    dev = describe_device(device)
    print(f'  device : {dev["device"]}' + (f' ({dev.get("gpu_name")})' if device.type == 'cuda' else ''))
    print(f'  tasks  : {tasks}  weights {weights}')
    if device.type != 'cuda':
        print('  [WARN] CUDA unavailable - running on CPU.')

    corpus, corpus_stats = build_corpus(verbose=True)
    graphs, graph_stats = build_graph_dataset(corpus, mode=args.features, verbose=True)

    strata = _stratum_labels(corpus)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    cfg = TrainConfig(
        tasks=tasks, task_weights=weights, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, hidden=args.hidden,
        dropout=args.dropout, layers=args.layers, patience=args.patience,
        seed=args.seed, amp=args.amp,
    )

    oof_rows: List[Dict[str, Any]] = []
    fold_train_rows: Dict[int, set] = {}
    fold_records: List[Dict[str, Any]] = []

    all_idx = np.arange(len(corpus))
    print(f'\n[KFOLD] {args.folds} stratified folds over {len(corpus)} molecules')

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(all_idx, strata), start=1):
        # Carve a validation slice from the training portion only.
        sub_strata = strata[trainval_idx]
        try:
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=args.val_fraction,
                random_state=args.seed, stratify=sub_strata,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=args.val_fraction, random_state=args.seed)

        fold_train_rows[fold] = set(int(i) for i in train_idx)

        train_graphs = subset_by_corpus_index(graphs, train_idx)
        val_graphs = subset_by_corpus_index(graphs, val_idx)
        test_graphs = subset_by_corpus_index(graphs, test_idx)

        model, info = train_one_model(
            train_graphs, val_graphs, corpus, train_idx, cfg, device,
            graph_stats['num_atom_features'], graph_stats['num_bond_features'],
        )
        te = predict_graphs(model, test_graphs, info['criterion'], device, tasks,
                            batch_size=args.batch_size)

        # Gather this fold's predictions into one wide row per molecule.
        per_mol: Dict[int, Dict[str, Any]] = {}
        fold_metrics: Dict[str, Any] = {}
        for t in tasks:
            if t not in te['per_task']:
                continue
            pack = te['per_task'][t]
            if weights[t] > 0:
                fold_metrics[t] = classification_metrics(
                    pack['y_true'], pack['y_pred'], pack['y_prob'], n_bootstraps=0)
            for row, yt, yp, pr in zip(pack['corpus_row'], pack['y_true'],
                                       pack['y_pred'], pack['y_prob']):
                rec = per_mol.setdefault(int(row), {'corpus_row': int(row), 'fold': fold})
                # Only record predictions from heads that were actually trained.
                if weights[t] > 0:
                    rec[f'{t}_pred'] = int(yp)
                    rec[f'{t}_prob'] = float(pr)
                rec[f'{t}_true'] = int(yt)

        oof_rows.extend(per_mol.values())

        summary = ' | '.join(
            f'{t} AUC={fold_metrics[t].get("ROC_AUC", float("nan")):.4f} '
            f'(n={fold_metrics[t]["n"]})'
            for t in fold_metrics
        )
        print(f'  fold {fold}/{args.folds}: train {info["train_molecules"]}, '
              f'val {len(val_graphs)}, test {len(test_graphs)}, '
              f'best epoch {info["best_epoch"]}  {summary}')

        fold_records.append({
            'fold': fold,
            'train_molecules': info['train_molecules'],
            'val_molecules': len(val_graphs),
            'test_molecules': len(test_graphs),
            'best_epoch': info['best_epoch'],
            'best_val_score': info['best_val_score'],
            'epochs_run': info['epochs_run'],
            'metrics': {t: {k: v for k, v in fold_metrics[t].items()
                            if not k.endswith('_CI95')} for t in fold_metrics},
        })

        del model
        empty_cuda_cache(device)

    oof = pd.DataFrame(oof_rows).sort_values('corpus_row').reset_index(drop=True)
    for t in ALL_TASKS:
        for col in (f'{t}_pred', f'{t}_prob', f'{t}_true'):
            if col not in oof.columns:
                oof[col] = np.nan

    integrity = assert_oof_integrity(corpus, oof, fold_train_rows, active)
    print(f'\n[CHECK] out-of-fold integrity passed: {integrity["molecules_scored"]} molecules '
          f'scored exactly once, no fold scored a molecule it trained on')
    print(f'[CHECK] dual-labelled molecules with both predictions: '
          f'{integrity["dual_labelled_with_both_predictions"]} of '
          f'{integrity["dual_labelled_total"]}')

    # Attach identifiers so the file is self-contained.
    meta = corpus.loc[oof['corpus_row'], ['inchikey', 'canonical_smiles',
                                          'y_bbb', 'm_bbb', 'y_pgp', 'm_pgp']].reset_index(drop=True)
    oof_out = pd.concat([oof.reset_index(drop=True), meta], axis=1)

    # Pooled out-of-fold metrics per task.
    pooled: Dict[str, Any] = {}
    print(f'\n[OOF] pooled out-of-fold metrics ({args.bootstraps} bootstraps)')
    for t in active:
        sel = oof_out[f'{t}_pred'].notna() & (oof_out[f'm_{t}'] == 1)
        if sel.sum() == 0:
            continue
        m = classification_metrics(
            oof_out.loc[sel, f'y_{t}'].values,
            oof_out.loc[sel, f'{t}_pred'].values,
            oof_out.loc[sel, f'{t}_prob'].values,
            n_bootstraps=args.bootstraps, seed=args.seed,
        )
        pooled[t] = m
        ci = m.get('ROC_AUC_CI95')
        ci_txt = f' [{ci[0]:.4f} - {ci[1]:.4f}]' if ci else ''
        print(f'  {t:>4}: n={m["n"]:<5} ROC-AUC={m["ROC_AUC"]:.4f}{ci_txt}  '
              f'MCC={m["MCC"]:.4f}  F1={m["F1_Score"]:.4f}')

    elapsed = time.time() - started
    oof_path = os.path.join(RESULTS_DIR, f'gnn_{run_id}_oof_predictions.csv')
    oof_out.to_csv(oof_path, index=False)

    report = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'arguments': vars(args),
        'tasks': tasks,
        'active_tasks': active,
        'task_weights': weights,
        'device': dev,
        'backend': backend,
        'corpus': corpus_stats,
        'label_independence': label_independence(corpus),
        'graphs': {k: v for k, v in graph_stats.items() if k != 'failed_inchikeys_sample'},
        'train_config': cfg.to_dict(),
        'oof_integrity': integrity,
        'folds': fold_records,
        'fold_summary': {t: summarise_folds([f['metrics'][t] for f in fold_records
                                             if t in f['metrics']]) for t in active},
        'pooled_oof_metrics': pooled,
        'elapsed_seconds': round(elapsed, 1),
        'cuda_memory': cuda_memory_summary(device),
    }
    report_path = os.path.join(RESULTS_DIR, f'gnn_{run_id}_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    print(f'\n[SAVE] {os.path.relpath(oof_path, BASE_DIR)}')
    print(f'[SAVE] {os.path.relpath(report_path, BASE_DIR)}')
    print(f'[TIME] {elapsed:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
