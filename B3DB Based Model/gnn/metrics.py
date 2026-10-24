"""
Classification Metric Suite with Bootstrap Confidence Intervals.

Deliberately mirrors the metric names and the 1,000-resample bootstrap protocol
used by scripts/run_validation.py for the XGBoost arm, so GNN and XGBoost results
drop into the same table without re-deriving anything.

Implemented here rather than imported from run_validation.py because that module
pulls in xgboost, matplotlib and seaborn at import time and runs filesystem setup
at module scope.
"""

from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix,
)

METRIC_NAMES = [
    'Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall_Sensitivity',
    'Specificity', 'F1_Score', 'ROC_AUC', 'PR_AUC', 'MCC', 'Brier_Score',
]


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    out: Dict[str, float] = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall_Sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': specificity,
        'F1_Score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Brier_Score': brier_score_loss(y_true, y_prob),
    }
    # ROC-AUC and PR-AUC are undefined on a single-class sample.
    if len(np.unique(y_true)) > 1:
        out['ROC_AUC'] = roc_auc_score(y_true, y_prob)
        out['PR_AUC'] = average_precision_score(y_true, y_prob)
    else:
        out['ROC_AUC'] = float('nan')
        out['PR_AUC'] = float('nan')
    out.update({'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)})
    return out


def classification_metrics(
    y_true, y_pred, y_prob,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full metric suite plus percentile bootstrap 95% confidence intervals.

    Set n_bootstraps=0 to skip the resampling (used for per-epoch validation,
    where only the point estimate is needed).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return {'n': 0, 'note': 'no labelled samples in this partition'}

    result: Dict[str, Any] = {'n': int(len(y_true))}
    result.update(_point_metrics(y_true, y_pred, y_prob))

    if n_bootstraps and n_bootstraps > 0:
        rng = np.random.RandomState(seed)
        records: Dict[str, List[float]] = {k: [] for k in METRIC_NAMES}
        n = len(y_true)
        for _ in range(n_bootstraps):
            idx = rng.randint(0, n, n)
            bt, bp, bq = y_true[idx], y_pred[idx], y_prob[idx]
            if len(np.unique(bt)) < 2:
                continue
            m = _point_metrics(bt, bp, bq)
            for k in METRIC_NAMES:
                if not np.isnan(m[k]):
                    records[k].append(m[k])

        for k in METRIC_NAMES:
            vals = records[k]
            if vals:
                result[f'{k}_CI95'] = [
                    round(float(np.percentile(vals, 2.5)), 4),
                    round(float(np.percentile(vals, 97.5)), 4),
                ]
        result['n_bootstraps'] = int(n_bootstraps)

    return result


def format_metrics_row(task: str, label: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens one metric dict into a CSV row with 'estimate [lo - hi]' strings."""
    row: Dict[str, Any] = {'Task': task, 'Evaluation': label, 'N': metrics.get('n', 0)}
    for k in METRIC_NAMES:
        if k not in metrics:
            continue
        est = metrics[k]
        row[k] = round(float(est), 4) if est == est else None  # NaN-safe
        ci = metrics.get(f'{k}_CI95')
        row[f'{k}_CI95'] = f'{ci[0]:.4f} - {ci[1]:.4f}' if ci else None
    for k in ('TP', 'FP', 'TN', 'FN'):
        if k in metrics:
            row[k] = metrics[k]
    return row


def summarise_folds(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean and standard deviation of each metric across cross-validation folds."""
    if not fold_metrics:
        return {}
    summary: Dict[str, Any] = {'n_folds': len(fold_metrics)}
    for k in METRIC_NAMES:
        vals = [m[k] for m in fold_metrics if k in m and m[k] == m[k]]
        if vals:
            summary[f'{k}_mean'] = round(float(np.mean(vals)), 4)
            summary[f'{k}_std'] = round(float(np.std(vals)), 4)
    return summary
