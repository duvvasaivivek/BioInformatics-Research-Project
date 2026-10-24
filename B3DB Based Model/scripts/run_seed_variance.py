"""
Seed-Variance, Minimum Detectable Effect, and P(A>B) for the Multi-Task Contrast.

The multi-task effects measured on a single seed are of order 1e-3 in ROC-AUC. A
single run cannot distinguish an effect that small from run-to-run noise, so the
null result it implies is not yet evidence. This script supplies the missing
statistics.

For each seed, the same seed drives BOTH the data split and the weight
initialisation, and all experiments share it. The multi-task-minus-single-task
contrast is therefore PAIRED by seed, which is what makes a paired t-test and a
minimum-detectable-effect calculation legitimate.

Reported per task:
  - mean and sd across seeds for each experiment
  - the paired delta (multi-task minus single-task) with a 95% CI
  - MDE = (t_{1-alpha/2,df} + t_{1-beta,df}) * sd_diff / sqrt(n_seeds)
  - P(A > B) over all cross-seed pairs, per Bouthillier et al.'s recommendation
    that benchmarks report this rather than comparing single numbers; their
    suggested decision threshold is 0.75.

This methodology is NOT a contribution of this work - a 2026 PROTAC-activity
preprint applies the same MDE-over-seeds reasoning and reports a +0.0025 AUROC
effect as "informative as a null result". It is adopted here and cited, not claimed.

Example
-------
    python scripts/run_seed_variance.py --seeds 42,43,44,45,46 --split random
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from itertools import product
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from gnn.device import (
    get_device, describe_device, tune_backend, cuda_memory_summary,
    reset_cuda_peak, empty_cuda_cache,
)
from gnn.data_assembly import build_corpus
from gnn.dataset import build_graph_dataset, subset_by_corpus_index
from gnn.splits import make_splits, assert_split_integrity
from gnn.training import TrainConfig, train_one_model, predict_graphs
from gnn.metrics import classification_metrics

RESULTS_DIR = os.path.join(BASE_DIR, 'results')

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    'single_bbb': {'tasks': ['bbb'], 'weights': {}},
    'single_pgp': {'tasks': ['pgp'], 'weights': {}},
    'multitask': {'tasks': ['bbb', 'pgp'], 'weights': {}},
}

HEADLINE = 'ROC_AUC'
SECONDARY = ['MCC', 'F1_Score', 'Balanced_Accuracy']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Seed variance and MDE for the multi-task contrast.')
    p.add_argument('--seeds', type=str, default='42,43,44,45,46')
    p.add_argument('--split', type=str, default='random', choices=('random', 'scaffold'))
    p.add_argument('--features', type=str, default='encoded', choices=('encoded', 'legacy'))
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--power', type=float, default=0.80)
    p.add_argument('--tag', type=str, default='')
    return p.parse_args()


def minimum_detectable_effect(diffs: np.ndarray, alpha: float, power: float) -> Dict[str, Any]:
    """
    Smallest paired effect this design could have detected.

    MDE = (t_{1-alpha/2, df} + t_{1-beta, df}) * sd / sqrt(n)

    An observed effect below the MDE means the study was not powered to detect it:
    the correct conclusion is "no effect detected at this sample size", never
    "no effect exists".
    """
    n = len(diffs)
    if n < 2:
        return {'n_seeds': int(n), 'note': 'need at least 2 seeds for an MDE'}
    df = n - 1
    sd = float(np.std(diffs, ddof=1))
    t_alpha = float(sps.t.ppf(1 - alpha / 2, df))
    t_beta = float(sps.t.ppf(power, df))
    mde = (t_alpha + t_beta) * sd / np.sqrt(n)
    return {
        'n_seeds': int(n),
        'sd_of_differences': round(sd, 6),
        't_alpha': round(t_alpha, 4),
        't_beta': round(t_beta, 4),
        'mde': round(float(mde), 6),
        'alpha': alpha,
        'power': power,
    }


def prob_a_beats_b(a: np.ndarray, b: np.ndarray) -> float:
    """
    P(A > B) over all cross pairs, with ties counted as half.

    Bouthillier et al. argue benchmarks should report this instead of comparing
    single numbers, and suggest 0.75 as a decision threshold.
    """
    pairs = list(product(a, b))
    wins = sum(1.0 if x > y else 0.5 if x == y else 0.0 for x, y in pairs)
    return float(wins / len(pairs)) if pairs else float('nan')


def main() -> int:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    tag = args.tag or f'seedvar_{args.split}'
    started = time.time()

    device = get_device()
    backend = tune_backend(device)
    reset_cuda_peak(device)

    print('=' * 92)
    print(f'SEED VARIANCE AND MINIMUM DETECTABLE EFFECT ({args.split} split)')
    print('=' * 92)
    dev = describe_device(device)
    print(f'  device : {dev["device"]}' + (f' ({dev.get("gpu_name")})' if device.type == 'cuda' else ''))
    print(f'  seeds  : {seeds}')
    print(f'  design : paired by seed - the same seed drives the split AND the init '
          f'for every experiment')

    corpus, _ = build_corpus(verbose=False)
    graphs, graph_stats = build_graph_dataset(corpus, mode=args.features, verbose=False)
    print(f'  corpus : {len(corpus)} molecules, '
          f'{graph_stats["num_atom_features"]} atom / {graph_stats["num_bond_features"]} bond features')

    records: List[Dict[str, Any]] = []

    for seed in seeds:
        splits = make_splits(corpus, mode=args.split, test_size=0.15, val_size=0.15, seed=seed)
        assert_split_integrity(corpus, splits, args.split)
        gsets = {k: subset_by_corpus_index(graphs, splits[k]) for k in ('train', 'val', 'test')}

        for exp_name, spec in EXPERIMENTS.items():
            cfg = TrainConfig(
                tasks=list(spec['tasks']), task_weights=dict(spec['weights']),
                epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden,
                layers=args.layers, patience=args.patience, seed=seed,
            )
            model, info = train_one_model(
                gsets['train'], gsets['val'], corpus, splits['train'], cfg, device,
                graph_stats['num_atom_features'], graph_stats['num_bond_features'],
            )
            te = predict_graphs(model, gsets['test'], info['criterion'], device,
                                cfg.tasks, batch_size=args.batch_size)

            for task in cfg.active_tasks():
                if task not in te['per_task']:
                    continue
                pack = te['per_task'][task]
                m = classification_metrics(pack['y_true'], pack['y_pred'], pack['y_prob'],
                                           n_bootstraps=0)
                rec = {'seed': seed, 'experiment': exp_name, 'task': task,
                       'n_test': m['n'], 'best_epoch': info['best_epoch']}
                rec.update({k: m[k] for k in [HEADLINE] + SECONDARY if k in m})
                records.append(rec)

            del model
            empty_cuda_cache(device)

        done = [r for r in records if r['seed'] == seed]
        detail = ', '.join(f'{r["experiment"]}/{r["task"]}={r[HEADLINE]:.4f}' for r in done)
        print(f'  seed {seed}: {detail}')

    df = pd.DataFrame(records)
    per_run_path = os.path.join(RESULTS_DIR, f'gnn_{tag}_per_run.csv')
    df.to_csv(per_run_path, index=False)

    # ------------------------------------------------------------- aggregation
    print('\n' + '-' * 92)
    print(f'ACROSS-SEED SUMMARY ({HEADLINE})')
    print('-' * 92)
    print(f'{"task":<6} {"experiment":<14} {"mean":>9} {"sd":>9} {"min":>9} {"max":>9} {"n":>4}')
    summary: List[Dict[str, Any]] = []
    for (task, exp), grp in df.groupby(['task', 'experiment']):
        vals = grp[HEADLINE].values
        summary.append({
            'task': task, 'experiment': exp, 'metric': HEADLINE,
            'mean': round(float(vals.mean()), 6), 'sd': round(float(vals.std(ddof=1)), 6),
            'min': round(float(vals.min()), 6), 'max': round(float(vals.max()), 6),
            'n_seeds': int(len(vals)),
        })
        print(f'{task:<6} {exp:<14} {vals.mean():>9.4f} {vals.std(ddof=1):>9.4f} '
              f'{vals.min():>9.4f} {vals.max():>9.4f} {len(vals):>4}')

    # --------------------------------------------- paired multi-task contrast
    print('\n' + '-' * 92)
    print('PAIRED CONTRAST: multi-task minus single-task')
    print('-' * 92)
    contrasts: List[Dict[str, Any]] = []

    for task in sorted(df['task'].unique()):
        single_name = f'single_{task}'
        a = df[(df['task'] == task) & (df['experiment'] == 'multitask')].set_index('seed')
        b = df[(df['task'] == task) & (df['experiment'] == single_name)].set_index('seed')
        shared = sorted(set(a.index) & set(b.index))
        if len(shared) < 2:
            print(f'  {task}: fewer than 2 paired seeds, skipping')
            continue

        for metric in [HEADLINE] + SECONDARY:
            if metric not in a.columns or metric not in b.columns:
                continue
            av = a.loc[shared, metric].values.astype(float)
            bv = b.loc[shared, metric].values.astype(float)
            diffs = av - bv
            mean_diff = float(diffs.mean())

            mde = minimum_detectable_effect(diffs, args.alpha, args.power)
            t_stat, p_val = sps.ttest_rel(av, bv)
            sem = float(sps.sem(diffs)) if len(diffs) > 1 else float('nan')
            crit = float(sps.t.ppf(1 - args.alpha / 2, len(diffs) - 1))
            ci = [round(mean_diff - crit * sem, 6), round(mean_diff + crit * sem, 6)]
            pab = prob_a_beats_b(av, bv)

            underpowered = bool(abs(mean_diff) < mde.get('mde', np.inf))
            verdict = ('no effect detected; study underpowered for an effect this small'
                       if underpowered else
                       ('multi-task better' if mean_diff > 0 else 'multi-task worse'))

            contrasts.append({
                'task': task, 'metric': metric, 'n_seeds': len(shared),
                'multitask_mean': round(float(av.mean()), 6),
                'single_mean': round(float(bv.mean()), 6),
                'mean_delta': round(mean_diff, 6),
                'delta_CI95_low': ci[0], 'delta_CI95_high': ci[1],
                'sd_of_differences': mde.get('sd_of_differences'),
                'mde': mde.get('mde'),
                'abs_delta_below_mde': underpowered,
                'paired_t': round(float(t_stat), 4),
                'paired_p': round(float(p_val), 4),
                'prob_multitask_beats_single': round(pab, 4),
                'exceeds_pab_threshold_0.75': bool(pab >= 0.75 or pab <= 0.25),
                'verdict': verdict,
            })

            if metric == HEADLINE:
                print(f'\n  {task.upper()}  ({len(shared)} paired seeds)')
                print(f'    multi-task {av.mean():.4f} vs single-task {bv.mean():.4f}')
                print(f'    delta            {mean_diff:+.5f}  95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]')
                print(f'    sd of diffs      {mde["sd_of_differences"]:.5f}')
                print(f'    MDE (80% power)  {mde["mde"]:.5f}')
                print(f'    |delta| < MDE    {underpowered}')
                print(f'    paired t-test    t={t_stat:+.3f}  p={p_val:.4f}')
                print(f'    P(multi > single) {pab:.3f}   (threshold 0.75)')
                print(f'    -> {verdict}')

    contrast_df = pd.DataFrame(contrasts)
    contrast_path = os.path.join(RESULTS_DIR, f'gnn_{tag}_contrasts.csv')
    contrast_df.to_csv(contrast_path, index=False)
    summary_path = os.path.join(RESULTS_DIR, f'gnn_{tag}_summary.csv')
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    elapsed = time.time() - started
    report = {
        'tag': tag,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'arguments': vars(args),
        'seeds': seeds,
        'device': dev,
        'backend': backend,
        'design': 'paired by seed; seed drives split and initialisation for all experiments',
        'summary': summary,
        'contrasts': contrasts,
        'methodology_note': (
            'MDE-over-seeds and P(A>B) reporting follow prior work (a 2026 PROTAC-activity '
            'preprint and Bouthillier et al. on variance in ML benchmarks) and are adopted '
            'here as a rigor layer, not claimed as a contribution.'
        ),
        'elapsed_seconds': round(elapsed, 1),
        'cuda_memory': cuda_memory_summary(device),
    }
    report_path = os.path.join(RESULTS_DIR, f'gnn_{tag}_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    print(f'\n[SAVE] {os.path.relpath(per_run_path, BASE_DIR)}')
    print(f'[SAVE] {os.path.relpath(summary_path, BASE_DIR)}')
    print(f'[SAVE] {os.path.relpath(contrast_path, BASE_DIR)}')
    print(f'[SAVE] {os.path.relpath(report_path, BASE_DIR)}')
    print(f'[TIME] {elapsed:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
