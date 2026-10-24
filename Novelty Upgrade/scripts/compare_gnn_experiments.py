"""
Consolidates Multi-Task GNN Experiment Reports into the Paper's Core Table.

Reads every gnn_*_report.json in results/ and emits a single comparison table of
single-task versus multi-task performance, per task and per split protocol, with
bootstrap confidence intervals and the single-to-multi delta.

The delta is the paper's central result. The pre-registered hypothesis is
asymmetric: the small P-gp task should gain from sharing an encoder with the large
BBB task, while the BBB task may not gain and may suffer negative transfer.
"""

import os
import sys
import json
import glob
from typing import Dict, List, Any, Optional

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')

HEADLINE_METRICS = ['ROC_AUC', 'PR_AUC', 'MCC', 'F1_Score', 'Balanced_Accuracy', 'Specificity']


def load_reports() -> List[Dict[str, Any]]:
    """
    Loads the per-run training reports from results/.

    The glob `gnn_*_report.json` also matches reports with other schemas written
    by sibling scripts (the decision-metric and seed-variance reports), so each
    candidate is checked for the training-report shape rather than assumed. A
    filename glob is not a schema.
    """
    reports = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, 'gnn_*_report.json'))):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError:
                print(f'[WARN] skipping unreadable {os.path.basename(path)}')
                continue
        if not isinstance(payload, dict):
            continue
        # Require exactly what build_table consumes: the run arguments and a
        # single-split test_metrics block. This deliberately excludes the K-fold
        # reports, which carry pooled out-of-fold metrics under a different key
        # and belong in run_metric_analysis.py, not in this single-split table.
        if 'arguments' not in payload or 'test_metrics' not in payload:
            continue
        reports.append(payload)
    return reports


def experiment_kind(report: Dict[str, Any]) -> str:
    """
    Run identity comes from the --tag, not from the active task list.

    Deriving it from active_tasks would make the pgp=0 ablation indistinguishable
    from the genuine single-task BBB baseline, and the task-weight sweep runs
    indistinguishable from the headline multi-task run.
    """
    tag = (report.get('arguments') or {}).get('tag') or ''
    if tag:
        return tag
    active = report.get('active_tasks', [])
    return 'multitask' if len(active) > 1 else f'single_{active[0]}' if active else 'unknown'


def build_table(reports: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rep in reports:
        kind = experiment_kind(rep)
        split = rep['arguments']['split']
        features = rep['arguments']['features']
        for task, metrics in rep.get('test_metrics', {}).items():
            if not isinstance(metrics, dict) or 'n' not in metrics:
                continue
            row: Dict[str, Any] = {
                'experiment': kind,
                'split': split,
                'features': features,
                'task': task,
                'n_test': metrics['n'],
                'best_epoch': rep.get('best_epoch'),
                'epochs_run': rep.get('epochs_run'),
                'params': rep['architecture']['trainable_parameters'],
                'train_seconds': rep.get('elapsed_seconds'),
            }
            for m in HEADLINE_METRICS:
                val = metrics.get(m)
                row[m] = round(float(val), 4) if isinstance(val, (int, float)) else None
                ci = metrics.get(f'{m}_CI95')
                row[f'{m}_CI95'] = f'[{ci[0]:.4f}, {ci[1]:.4f}]' if ci else None
            rows.append(row)
    return pd.DataFrame(rows)


def add_deltas(table: pd.DataFrame) -> pd.DataFrame:
    """
    Attaches the single-task -> multi-task delta for each (task, split, features).

    Pairs only the two headline runs ('single_<task>' and 'multitask'); the
    task-weight sweep and the pgp=0 ablation are reported separately rather than
    averaged into the comparison.
    """
    deltas: List[Dict[str, Any]] = []
    for (task, split, features), grp in table.groupby(['task', 'split', 'features']):
        single = grp[grp['experiment'] == f'single_{task}']
        multi = grp[grp['experiment'] == 'multitask']
        if single.empty or multi.empty:
            continue
        row: Dict[str, Any] = {
            'task': task, 'split': split, 'features': features,
            'n_test': int(multi.iloc[0]['n_test']),
        }
        for m in HEADLINE_METRICS:
            s_val, m_val = single.iloc[0][m], multi.iloc[0][m]
            if s_val is None or m_val is None:
                continue
            row[f'single_{m}'] = s_val
            row[f'multi_{m}'] = m_val
            row[f'delta_{m}'] = round(float(m_val) - float(s_val), 4)
        deltas.append(row)
    return pd.DataFrame(deltas)


def ci_overlaps(a: Optional[str], b: Optional[str]) -> Optional[bool]:
    """True when two '[lo, hi]' interval strings overlap, i.e. the delta is not clear-cut."""
    if not a or not b:
        return None
    try:
        alo, ahi = [float(x) for x in a.strip('[]').split(',')]
        blo, bhi = [float(x) for x in b.strip('[]').split(',')]
    except ValueError:
        return None
    return not (ahi < blo or bhi < alo)


def main() -> int:
    reports = load_reports()
    if not reports:
        print(f'[WARN] No gnn_*_report.json found in {RESULTS_DIR}. Run training first.')
        return 1

    table = build_table(reports)
    deltas = add_deltas(table)

    table_path = os.path.join(RESULTS_DIR, 'gnn_experiment_comparison.csv')
    table.sort_values(['task', 'split', 'experiment']).to_csv(table_path, index=False)

    print('=' * 100)
    print('MULTI-TASK GNN: SINGLE-TASK vs MULTI-TASK')
    print('=' * 100)

    for split in sorted(table['split'].unique()):
        print(f'\n--- {split.upper()} SPLIT ---')
        sub = table[table['split'] == split]
        print(f'{"task":<5} {"experiment":<14} {"n":>5} {"ROC-AUC":>9} {"95% CI":>18} '
              f'{"MCC":>8} {"F1":>8} {"BalAcc":>8}')
        for task in sorted(sub['task'].unique()):
            for _, r in sub[sub['task'] == task].sort_values('experiment').iterrows():
                print(f'{r["task"]:<5} {r["experiment"]:<14} {r["n_test"]:>5} '
                      f'{r["ROC_AUC"]:>9.4f} {str(r["ROC_AUC_CI95"]):>18} '
                      f'{r["MCC"]:>8.4f} {r["F1_Score"]:>8.4f} {r["Balanced_Accuracy"]:>8.4f}')

    if not deltas.empty:
        delta_path = os.path.join(RESULTS_DIR, 'gnn_multitask_delta.csv')
        deltas.to_csv(delta_path, index=False)

        print('\n' + '=' * 100)
        print('MULTI-TASK EFFECT (multi-task minus single-task)')
        print('=' * 100)
        print(f'{"task":<5} {"split":<10} {"d ROC-AUC":>10} {"d MCC":>9} {"d F1":>9} '
              f'{"d BalAcc":>9}  verdict')
        for _, r in deltas.sort_values(['task', 'split']).iterrows():
            d_auc = r.get('delta_ROC_AUC')
            # A delta whose confidence intervals overlap is within noise.
            srow = table[(table['task'] == r['task']) & (table['split'] == r['split'])
                         & (table['experiment'] == f'single_{r["task"]}')]
            mrow = table[(table['task'] == r['task']) & (table['split'] == r['split'])
                         & (table['experiment'] == 'multitask')]
            overlap = ci_overlaps(
                srow.iloc[0]['ROC_AUC_CI95'] if not srow.empty else None,
                mrow.iloc[0]['ROC_AUC_CI95'] if not mrow.empty else None,
            )
            if overlap is None:
                verdict = 'incomparable'
            elif overlap:
                verdict = 'within noise (CIs overlap)'
            else:
                verdict = 'improved' if d_auc > 0 else 'degraded'
            print(f'{r["task"]:<5} {r["split"]:<10} {d_auc:>+10.4f} '
                  f'{r.get("delta_MCC", float("nan")):>+9.4f} '
                  f'{r.get("delta_F1_Score", float("nan")):>+9.4f} '
                  f'{r.get("delta_Balanced_Accuracy", float("nan")):>+9.4f}  {verdict}')

        print(f'\n[SAVE] {os.path.relpath(delta_path, BASE_DIR)}')

    print(f'[SAVE] {os.path.relpath(table_path, BASE_DIR)}')

    # Scaffold-vs-random degradation, the asymmetry claim from the plan.
    print('\n' + '=' * 100)
    print('CHEMOTYPE TRANSFER (random -> scaffold, same experiment)')
    print('=' * 100)
    for exp in sorted(table['experiment'].unique()):
        for task in sorted(table['task'].unique()):
            rnd = table[(table['experiment'] == exp) & (table['task'] == task)
                        & (table['split'] == 'random')]
            scf = table[(table['experiment'] == exp) & (table['task'] == task)
                        & (table['split'] == 'scaffold')]
            if rnd.empty or scf.empty:
                continue
            d = scf.iloc[0]['ROC_AUC'] - rnd.iloc[0]['ROC_AUC']
            print(f'  {exp:<14} {task:<5} ROC-AUC {rnd.iloc[0]["ROC_AUC"]:.4f} -> '
                  f'{scf.iloc[0]["ROC_AUC"]:.4f}  ({d:+.4f})')

    return 0


if __name__ == '__main__':
    sys.exit(main())
