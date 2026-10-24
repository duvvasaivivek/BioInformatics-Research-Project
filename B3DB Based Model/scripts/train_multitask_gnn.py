"""
Multi-Task GNN Training: Joint BBB Permeability and P-glycoprotein Efflux.

Trains a shared GINEConv encoder with one classification head per barrier, on the
merged B3DB + Broccatelli corpus, with masked per-task loss. Runs on CUDA when
available.

The three experiments the paper compares:

    single-task BBB     --tasks bbb
    single-task P-gp    --tasks pgp
    multi-task          --tasks bbb,pgp

Splits are computed ONCE at molecule level over the whole corpus and reused by all
three, so each task's test set is identical across experiments and the comparison
is apples to apples.

Examples
--------
    python scripts/train_multitask_gnn.py --tasks bbb,pgp --split random
    python scripts/train_multitask_gnn.py --tasks pgp --split scaffold --tag pgp_only
    python scripts/train_multitask_gnn.py --tasks bbb,pgp --task-weights pgp=0 --tag ablate_pgp
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from gnn.device import (
    get_device, describe_device, tune_backend, cuda_memory_summary,
    reset_cuda_peak, empty_cuda_cache,
)
from gnn.data_assembly import build_corpus, label_independence
from gnn.dataset import build_graph_dataset, subset_by_corpus_index, make_loader
from gnn.splits import make_splits, assert_split_integrity
from gnn.model import MultiTaskGNN
from gnn.losses import MaskedMultiTaskLoss, build_class_weights
from gnn.metrics import classification_metrics, format_metrics_row
# Training helpers live in gnn/training.py so this runner, the K-fold runner and
# the seed-variance runner share one implementation instead of three copies.
from gnn.training import (
    set_all_seeds, parse_task_weights, move_batch, run_epoch, quick_f1,
)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ALL_TASKS = ('bbb', 'pgp')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train the dual-barrier multi-task GNN.')
    p.add_argument('--tasks', type=str, default='bbb,pgp',
                   help='Comma-separated active tasks: bbb, pgp, or bbb,pgp')
    p.add_argument('--task-weights', type=str, default='',
                   help='Override loss weights, e.g. "pgp=0" or "bbb=1,pgp=2"')
    p.add_argument('--split', type=str, default='random', choices=('random', 'scaffold'))
    p.add_argument('--features', type=str, default='encoded', choices=('encoded', 'legacy'))
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.30)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--clip-grad', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-size', type=float, default=0.15)
    p.add_argument('--val-size', type=float, default=0.15)
    p.add_argument('--bootstraps', type=int, default=1000)
    p.add_argument('--residual', action='store_true', help='Enable skip connections (off by default)')
    p.add_argument('--amp', action='store_true',
                   help='Mixed-precision training. Off by default: it changes numerics slightly, '
                        'and these runs are small enough that determinism is worth more than speed.')
    p.add_argument('--no-cache', action='store_true', help='Rebuild the graph cache')
    p.add_argument('--tag', type=str, default='', help='Label for output filenames')
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    for t in tasks:
        if t not in ALL_TASKS:
            raise ValueError(f'Unknown task {t!r}; expected from {ALL_TASKS}')
    task_weights = parse_task_weights(args.task_weights, tasks)
    active_tasks = [t for t in tasks if task_weights[t] > 0]
    if not active_tasks:
        raise ValueError('All task weights are zero; nothing to train.')

    tag = args.tag or ('multitask' if len(active_tasks) > 1 else f'single_{active_tasks[0]}')
    run_id = f'{tag}_{args.split}_{args.features}'
    started = time.time()

    set_all_seeds(args.seed)
    device = get_device()
    backend = tune_backend(device)
    reset_cuda_peak(device)

    print('=' * 78)
    print('DUAL-BARRIER MULTI-TASK GNN')
    print('=' * 78)
    dev_info = describe_device(device)
    for k, v in dev_info.items():
        print(f'  {k:<24}: {v}')
    print(f'  {"tf32 / cudnn.benchmark":<24}: {backend["tf32_enabled"]} / {backend["cudnn_benchmark"]}')
    if device.type != 'cuda':
        print('  [WARN] CUDA not available - running on CPU, this will be slow.')
    print(f'  {"run":<24}: {run_id}')
    print(f'  {"tasks / weights":<24}: {tasks} / {task_weights}')
    print()

    # ---------------- data ----------------
    corpus, corpus_stats = build_corpus(verbose=not args.quiet)
    independence = label_independence(corpus)
    graphs, graph_stats = build_graph_dataset(
        corpus, mode=args.features, use_cache=not args.no_cache, verbose=not args.quiet
    )

    splits = make_splits(
        corpus, mode=args.split, test_size=args.test_size,
        val_size=args.val_size, seed=args.seed,
    )
    split_report = assert_split_integrity(corpus, splits, args.split)
    print(f'\n[SPLIT] {args.split}: '
          f'train {split_report["sizes"]["train"]} / '
          f'val {split_report["sizes"]["val"]} / '
          f'test {split_report["sizes"]["test"]}  '
          f'(integrity checks passed)')
    for t in tasks:
        tc = split_report['per_task_counts']
        print(f'[SPLIT]   {t}: '
              + ' | '.join(f'{n} {tc[n][t]["labelled"]} ({tc[n][t]["positive"]}+/{tc[n][t]["negative"]}-)'
                           for n in ('train', 'val', 'test')))

    graph_sets = {name: subset_by_corpus_index(graphs, splits[name]) for name in ('train', 'val', 'test')}

    # Molecules with no label for any active task contribute exactly zero loss;
    # dropping them from TRAIN only is a speed optimisation, never a change of
    # evaluation. Val and test stay complete so metrics cover every labelled molecule.
    def has_active_label(g) -> bool:
        return any(float(getattr(g, f'm_{t}').item()) > 0 for t in active_tasks)

    before = len(graph_sets['train'])
    graph_sets['train'] = [g for g in graph_sets['train'] if has_active_label(g)]
    dropped = before - len(graph_sets['train'])
    if dropped:
        print(f'[SPLIT]   dropped {dropped} train molecules with no label for the active tasks')

    loaders = {
        'train': make_loader(graph_sets['train'], args.batch_size, shuffle=True, device=device),
        'val': make_loader(graph_sets['val'], args.batch_size, shuffle=False, device=device),
        'test': make_loader(graph_sets['test'], args.batch_size, shuffle=False, device=device),
    }

    # ---------------- model ----------------
    atom_dim = graph_stats['num_atom_features']
    bond_dim = graph_stats['num_bond_features']
    model = MultiTaskGNN(
        num_atom_features=atom_dim,
        num_bond_features=bond_dim,
        tasks=tasks,
        hidden_channels=args.hidden,
        dropout=args.dropout,
        num_layers=args.layers,
        residual=args.residual,
    ).to(device)

    class_weights = build_class_weights(corpus, tasks, splits['train'], device)
    criterion = MaskedMultiTaskLoss(tasks, class_weights, task_weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler(device.type) if (args.amp and device.type == 'cuda') else None

    print(f'\n[MODEL] {model.count_parameters():,} trainable parameters '
          f'({atom_dim} atom / {bond_dim} bond features, hidden {args.hidden}, {args.layers} layers)')
    print(f'[MODEL] class weights: '
          + ', '.join(f'{t}={class_weights[t].cpu().numpy().round(3).tolist()}' for t in class_weights))
    print(f'[MODEL] mixed precision: {"on" if scaler else "off"}')

    # ---------------- training ----------------
    best_score = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    stale = 0
    history: List[Dict[str, Any]] = []

    print(f'\n[TRAIN] up to {args.epochs} epochs, early stopping patience {args.patience}')
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, loaders['train'], criterion, device, tasks,
                       optimizer=optimizer, scaler=scaler, clip_grad=args.clip_grad)
        va = run_epoch(model, loaders['val'], criterion, device, tasks)

        # Early stopping on the mean validation F1 across ACTIVE tasks. The
        # reference implementation drives scheduling, checkpointing and stopping
        # from a single scalar; with two tasks that scalar has to be explicit.
        task_f1 = {t: quick_f1(va['per_task'][t]) for t in active_tasks if t in va['per_task']}
        score = float(np.mean(list(task_f1.values()))) if task_f1 else 0.0

        scheduler.step(score)
        history.append({
            'epoch': epoch,
            'train_loss': round(tr['loss'], 5),
            'val_loss': round(va['loss'], 5),
            'val_score': round(score, 5),
            **{f'val_f1_{t}': round(v, 5) for t, v in task_f1.items()},
            'lr': optimizer.param_groups[0]['lr'],
        })

        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if not args.quiet and (epoch % 10 == 0 or epoch == 1):
            detail = ' '.join(f'{t}_f1={v:.4f}' for t, v in task_f1.items())
            print(f'  epoch {epoch:>3}  train_loss={tr["loss"]:.4f}  '
                  f'val_loss={va["loss"]:.4f}  {detail}  score={score:.4f}')

        if stale >= args.patience:
            print(f'  early stop at epoch {epoch} (no improvement for {args.patience} epochs)')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f'[TRAIN] best epoch {best_epoch}, val score {best_score:.4f}')

    # ---------------- test ----------------
    te = run_epoch(model, loaders['test'], criterion, device, tasks)
    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []

    print(f'\n[TEST] {args.split} split, {args.bootstraps} bootstrap resamples')
    for t in tasks:
        if t not in te['per_task']:
            continue
        pack = te['per_task'][t]
        m = classification_metrics(pack['y_true'], pack['y_pred'], pack['y_prob'],
                                   n_bootstraps=args.bootstraps, seed=args.seed)
        results[t] = m
        rows.append(format_metrics_row(t, f'{run_id}', m))
        auc = m.get('ROC_AUC', float('nan'))
        ci = m.get('ROC_AUC_CI95')
        ci_txt = f' [{ci[0]:.4f} - {ci[1]:.4f}]' if ci else ''
        print(f'  {t:>4}: n={m["n"]:<5} ROC-AUC={auc:.4f}{ci_txt}  '
              f'MCC={m.get("MCC", float("nan")):.4f}  F1={m.get("F1_Score", float("nan")):.4f}  '
              f'BalAcc={m.get("Balanced_Accuracy", float("nan")):.4f}')

        # Persist per-molecule test predictions. No per-molecule predictions were
        # saved anywhere in this repository before now, which blocked any
        # error-stratification analysis.
        idx = pack['corpus_row']
        pred_frames.append(pd.DataFrame({
            'task': t,
            'inchikey': corpus.iloc[idx]['inchikey'].values,
            'canonical_smiles': corpus.iloc[idx]['canonical_smiles'].values,
            'y_true': pack['y_true'],
            'y_pred': pack['y_pred'],
            'y_prob': pack['y_prob'],
            'm_bbb': corpus.iloc[idx]['m_bbb'].values,
            'y_bbb': corpus.iloc[idx]['y_bbb'].values,
            'm_pgp': corpus.iloc[idx]['m_pgp'].values,
            'y_pgp': corpus.iloc[idx]['y_pgp'].values,
        }))

    # ---------------- persistence ----------------
    elapsed = time.time() - started
    mem = cuda_memory_summary(device)

    record: Dict[str, Any] = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'arguments': vars(args),
        'tasks': tasks,
        'active_tasks': active_tasks,
        'task_weights': task_weights,
        'device': dev_info,
        'backend': backend,
        'corpus': corpus_stats,
        'label_independence': independence,
        'graphs': {k: v for k, v in graph_stats.items() if k != 'failed_inchikeys_sample'},
        'split_report': split_report,
        'architecture': model.architecture_metadata(),
        'best_epoch': best_epoch,
        'best_val_score': round(float(best_score), 5),
        'epochs_run': len(history),
        'test_metrics': {t: {k: v for k, v in results[t].items()} for t in results},
        'elapsed_seconds': round(elapsed, 1),
        'cuda_memory': mem,
    }

    json_path = os.path.join(RESULTS_DIR, f'gnn_{run_id}_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, default=str)

    csv_path = os.path.join(RESULTS_DIR, f'gnn_{run_id}_metrics.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    hist_path = os.path.join(RESULTS_DIR, f'gnn_{run_id}_history.csv')
    pd.DataFrame(history).to_csv(hist_path, index=False)

    if pred_frames:
        pred_path = os.path.join(RESULTS_DIR, f'gnn_{run_id}_test_predictions.csv')
        pd.concat(pred_frames, ignore_index=True).to_csv(pred_path, index=False)

    ckpt_path = os.path.join(MODELS_DIR, f'gnn_{run_id}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': model.architecture_metadata(),
        'featurization': graph_stats['featurization'],
        'run_id': run_id,
        'best_epoch': best_epoch,
        'best_val_score': float(best_score),
        'task_weights': task_weights,
    }, ckpt_path)

    print(f'\n[SAVE] {os.path.relpath(json_path, BASE_DIR)}')
    print(f'[SAVE] {os.path.relpath(csv_path, BASE_DIR)}')
    print(f'[SAVE] {os.path.relpath(hist_path, BASE_DIR)}')
    if pred_frames:
        print(f'[SAVE] {os.path.relpath(pred_path, BASE_DIR)}')
    print(f'[SAVE] {os.path.relpath(ckpt_path, BASE_DIR)}')
    if mem:
        print(f'[GPU ] peak allocated {mem["peak_allocated_gb"]} GB / reserved {mem["peak_reserved_gb"]} GB')
    print(f'[TIME] {elapsed:.1f}s')

    empty_cuda_cache(device)
    return 0


if __name__ == '__main__':
    sys.exit(main())
