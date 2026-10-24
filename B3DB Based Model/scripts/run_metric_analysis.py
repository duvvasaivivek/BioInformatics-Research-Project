"""
Decision-Aware Metric Analysis on Out-of-Fold Predictions.

Scores three ways of making the dual-barrier CNS decision on the ~216 molecules
that carry both ground-truth labels:

  A. BBB-only            - a single-axis permeability model. It has no efflux
                           output, so scoring it means assuming P-gp-negative for
                           everything, which is exactly what a BBB-only pipeline
                           implies in practice. This is the status quo in the
                           published BBB literature.
  B. Composed two-model  - independent single-task BBB and P-gp models, their
                           predictions combined per molecule. Valid because both
                           K-fold runs use the same seed and the same
                           stratification, so the fold assignment is identical and
                           both predictions are genuinely out-of-fold.
  C. Multi-task          - one shared-encoder model with two heads.

Metrics M1-M5 come from gnn/decision_metrics.py. The comparison that matters is
B versus C: whether sharing an encoder buys anything on the joint decision, as
opposed to on either axis alone.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from gnn.decision_metrics import (
    full_decision_report, flatten_decision_report, DEFAULT_LAMBDAS,
)
from gnn.metrics import classification_metrics

RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def oof_path(tag: str, folds: int, features: str) -> str:
    return os.path.join(RESULTS_DIR, f'gnn_{tag}_kfold{folds}_{features}_oof_predictions.csv')


def load_oof(tag: str, folds: int, features: str) -> Optional[pd.DataFrame]:
    path = oof_path(tag, folds, features)
    if not os.path.exists(path):
        print(f'[WARN] missing {os.path.relpath(path, BASE_DIR)}')
        return None
    return pd.read_csv(path)


def dual_subset(df: pd.DataFrame) -> pd.DataFrame:
    """The molecules with ground truth on both axes."""
    return df[(df['m_bbb'] == 1) & (df['m_pgp'] == 1)].copy()


def main() -> int:
    ap = argparse.ArgumentParser(description='Decision-aware metrics on OOF predictions.')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--features', type=str, default='encoded')
    ap.add_argument('--bootstraps', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    single_bbb = load_oof('single_bbb', args.folds, args.features)
    single_pgp = load_oof('single_pgp', args.folds, args.features)
    multitask = load_oof('multitask', args.folds, args.features)

    if multitask is None or single_bbb is None:
        print('[ERROR] need at least the single_bbb and multitask OOF files. '
              'Run scripts/run_kfold_multitask.py first.')
        return 1

    print('=' * 92)
    print('DECISION-AWARE DUAL-BARRIER METRICS (out-of-fold)')
    print('=' * 92)

    evaluations: List[Tuple[str, pd.DataFrame, str]] = []

    # ---- A. BBB-only ----
    a = dual_subset(single_bbb)
    a['pgp_pred_used'] = 0  # single-axis model implies "not a substrate"
    a['bbb_pred_used'] = a['bbb_pred']
    a['bbb_prob_used'] = a['bbb_prob']
    evaluations.append(('A_bbb_only', a, 'single-axis BBB model, P-gp assumed negative'))

    # ---- B. Composed two-model ----
    if single_pgp is not None:
        merged = single_bbb.merge(
            single_pgp[['corpus_row', 'pgp_pred', 'pgp_prob', 'fold']],
            on='corpus_row', how='inner', suffixes=('', '_pgpmodel'),
        )
        # Both runs must have produced the same fold assignment, or the composed
        # predictions are not from comparable out-of-fold models.
        same_folds = bool((merged['fold'] == merged['fold_pgpmodel']).all())
        print(f'\n[CHECK] single-task runs share an identical fold assignment: {same_folds}')
        if not same_folds:
            n_diff = int((merged['fold'] != merged['fold_pgpmodel']).sum())
            print(f'[WARN] {n_diff} molecules differ in fold between the two single-task runs; '
                  f'the composed baseline is not strictly comparable and is reported with that caveat.')
        b = dual_subset(merged)
        b['bbb_pred_used'] = b['bbb_pred']
        b['bbb_prob_used'] = b['bbb_prob']
        b['pgp_pred_used'] = b['pgp_pred_pgpmodel']
        evaluations.append(('B_composed_two_model', b,
                            'independent single-task BBB + P-gp models, composed'))

    # ---- C. Multi-task ----
    c = dual_subset(multitask)
    c['bbb_pred_used'] = c['bbb_pred']
    c['bbb_prob_used'] = c['bbb_prob']
    c['pgp_pred_used'] = c['pgp_pred']
    evaluations.append(('C_multitask', c, 'shared-encoder multi-task model'))

    reports: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    for name, df, description in evaluations:
        df = df.dropna(subset=['bbb_pred_used', 'pgp_pred_used'])
        rep = full_decision_report(
            y_bbb_true=df['y_bbb'].astype(int).values,
            y_pgp_true=df['y_pgp'].astype(int).values,
            y_bbb_pred=df['bbb_pred_used'].astype(int).values,
            y_pgp_pred=df['pgp_pred_used'].astype(int).values,
            y_bbb_prob=df['bbb_prob_used'].astype(float).values,
            n_bootstraps=args.bootstraps, seed=args.seed, lambdas=DEFAULT_LAMBDAS,
        )
        rep['description'] = description
        reports[name] = rep
        row = flatten_decision_report(name, rep)
        row['Description'] = description
        rows.append(row)

    # ------------------------------------------------------------------ output
    q_support = reports[evaluations[0][0]]['quadrant']['quadrant_support']
    print(f'\nGround-truth quadrant composition (n={reports[evaluations[0][0]]["quadrant"]["n"]}):')
    for q, n in q_support.items():
        print(f'  {q}: {n}')

    print('\n' + '-' * 92)
    print('M1-M3  QUADRANT DECISION')
    print('-' * 92)
    print(f'{"evaluation":<24} {"exact-match":>12} {"kappa":>9} {"paradox-rec":>12} '
          f'{"paradox-prec":>13} {"leakage":>9}')
    for name, _, _ in evaluations:
        q = reports[name]['quadrant']
        print(f'{name:<24} {q["quadrant_exact_match"]:>12.4f} {q["quadrant_kappa"]:>9.4f} '
              f'{q["paradox_recall"]:>12.4f} {q["paradox_precision"]:>13.4f} '
              f'{q["paradox_leakage_rate"]:>9.4f}')

    print('\nwith 95% bootstrap intervals:')
    for name, _, _ in evaluations:
        q = reports[name]['quadrant']
        em = q.get('quadrant_exact_match_CI95')
        pr = q.get('paradox_recall_CI95')
        print(f'  {name:<24} exact-match {q["quadrant_exact_match"]:.4f} '
              f'[{em[0]:.4f}, {em[1]:.4f}]' if em else f'  {name}')
        if pr:
            print(f'  {"":<24} paradox-rec {q["paradox_recall"]:.4f} [{pr[0]:.4f}, {pr[1]:.4f}]')

    print('\n' + '-' * 92)
    print('M3  PARADOX LEAKAGE - true Q2 (BBB+/P-gp+) molecules progressed as Q1')
    print('-' * 92)
    for name, _, _ in evaluations:
        q = reports[name]['quadrant']
        print(f'  {name:<24} {int(q["paradox_leaked_n"]):>3} of {int(q["paradox_true_n"]):>3} '
              f'leaked  ({q["paradox_leakage_rate"]:.1%})')

    print('\n' + '-' * 92)
    print('M4  EFFLUX-DISPARITY GAP - BBB AUC on non-substrates minus substrates')
    print('-' * 92)
    for name, _, _ in evaluations:
        e = reports[name].get('efflux_disparity', {})
        if 'efflux_disparity_gap' not in e:
            print(f'  {name:<24} undefined - {e.get("note", "")}')
            continue
        ci = e.get('efflux_disparity_gap_CI95')
        ci_txt = f'[{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else ''
        verdict = 'gap detected' if e.get('gap_excludes_zero') else 'no gap detected (CI spans 0)'
        print(f'  {name:<24} AUC {e["auc_non_substrate"]:.4f} (n={e["n_non_substrate"]}) vs '
              f'{e["auc_substrate"]:.4f} (n={e["n_substrate"]})  '
              f'gap {e["efflux_disparity_gap"]:+.4f} {ci_txt}  {verdict}')

    print('\n' + '-' * 92)
    print('M5  EXPECTED DECISION REGRET vs COST RATIO lambda')
    print('-' * 92)
    header = ' '.join(f'{lam:>8g}' for lam in DEFAULT_LAMBDAS)
    print(f'{"evaluation":<24} {header}')
    for name, _, _ in evaluations:
        r = reports[name]['regret']
        vals = ' '.join(f'{pt["regret"]:>8.4f}' for pt in r['regret_curve'])
        print(f'{name:<24} {vals}')
    print('\n  false progressions by true quadrant (molecules wrongly advanced as Q1):')
    for name, _, _ in evaluations:
        r = reports[name]['regret']
        print(f'    {name:<24} {r["false_progress_by_true_quadrant"]}  '
              f'(of which paradoxical: {r["paradoxical_false_progress"]})')

    # Where does adding the efflux axis start to pay for itself? Regret is linear
    # in lambda, so two methods cross at a single point that can be solved exactly.
    # Below it the cheaper-to-progress single-axis rule wins; above it the two-axis
    # rule does. Reporting the crossover is more honest than picking a lambda.
    base = reports.get('A_bbb_only', {}).get('regret')
    if base is not None:
        print('\n  cost-ratio crossover against the single-axis baseline:')
        a_fp, a_fn = base['n_false_progress'], base['n_missed_opportunity']
        for name, _, _ in evaluations:
            if name == 'A_bbb_only':
                continue
            r = reports[name]['regret']
            den = a_fp - r['n_false_progress']
            if den == 0:
                print(f'    {name:<24} no crossover (identical false-progression count)')
                continue
            lam = (r['n_missed_opportunity'] - a_fn) / den
            print(f'    {name:<24} crossover at lambda = {lam:.3f}  '
                  f'(below this, the single-axis rule is preferable)')
            reports[name]['regret']['crossover_lambda_vs_bbb_only'] = round(float(lam), 4)

    # Per-axis OOF metrics on the same dual-labelled subset, for context.
    print('\n' + '-' * 92)
    print('CONTEXT  per-axis metrics on the same dual-labelled subset')
    print('-' * 92)
    for name, df, _ in evaluations:
        d = df.dropna(subset=['bbb_pred_used', 'pgp_pred_used'])
        bbb = classification_metrics(d['y_bbb'].astype(int), d['bbb_pred_used'].astype(int),
                                     d['bbb_prob_used'].astype(float), n_bootstraps=0)
        print(f'  {name:<24} BBB  AUC={bbb["ROC_AUC"]:.4f}  MCC={bbb["MCC"]:.4f}  n={bbb["n"]}')

    csv_path = os.path.join(RESULTS_DIR, 'gnn_decision_metrics.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path = os.path.join(RESULTS_DIR, 'gnn_decision_metrics_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2, default=str)

    print(f'\n[SAVE] {os.path.relpath(csv_path, BASE_DIR)}')
    print(f'[SAVE] {os.path.relpath(json_path, BASE_DIR)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
