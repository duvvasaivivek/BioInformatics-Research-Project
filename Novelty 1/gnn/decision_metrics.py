"""
Decision-Aware Dual-Barrier Evaluation Metrics.

Standard ADMET evaluation scores each endpoint separately: a model can report
ROC-AUC 0.94 on blood-brain barrier permeability and 0.94 on P-glycoprotein efflux
while still placing a large fraction of molecules in the wrong therapeutic quadrant.
Nothing in the current literature evaluates the dual-barrier decision jointly.

Five metrics, all requiring molecules with BOTH ground-truth labels:

  M1 quadrant_exact_match  - both axes correct (subset accuracy)
  M2 quadrant_kappa        - chance-corrected 4-class agreement + 4x4 matrix
  M3 paradox_*             - recall/precision/F1 on Q2 (BBB+/P-gp+), plus the
                             paradox leakage rate: true-Q2 molecules called Q1
  M4 efflux_disparity_gap  - AUC(non-substrates) - AUC(substrates) for the BBB task
  M5 decision_regret       - expected cost over a progression decision, as a curve
                             in the cost ratio lambda

The mathematics is standard. The contribution is the framing: these are the numbers
that matter for the CNS progression decision, and none of them is currently reported.

Honest limits, which belong in any write-up that uses these:
  - Dual labels exist for only ~216 molecules, so every interval here is wide.
  - Q4 (BBB-/P-gp+) has ~19 molecules; its conditional numbers are indicative only.
  - A single-axis BBB model scores exactly 0 on paradox recall by construction. That
    is a statement about what single-axis evaluation can express, not a fair contest.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score,
)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from core.decision_matrix import quadrant_id, QUADRANT_IDS, QUADRANT_NAMES

PARADOX_QUADRANT = 2       # BBB+ / P-gp+  - permeant but effluxed
PROGRESS_QUADRANT = 1      # BBB+ / P-gp-  - the quadrant a project would progress

DEFAULT_LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def assign_quadrants(y_bbb: Sequence[int], y_pgp: Sequence[int]) -> np.ndarray:
    """Vectorised quadrant assignment, delegating the convention to core.decision_matrix."""
    y_bbb = np.asarray(y_bbb).astype(int)
    y_pgp = np.asarray(y_pgp).astype(int)
    if len(y_bbb) != len(y_pgp):
        raise ValueError('BBB and P-gp arrays must be the same length')
    return np.array([quadrant_id(b, p) for b, p in zip(y_bbb, y_pgp)], dtype=int)


def _percentile_ci(values: List[float]) -> Optional[List[float]]:
    """
    Percentile bootstrap interval, stored at 6 decimals.

    Not 4: the multi-task effects under study are of order 1e-3, and rounding
    stored values to 4 dp makes exact delta arithmetic impossible. Display
    formatting happens in _fmt_ci.
    """
    if not values:
        return None
    return [round(float(np.percentile(values, 2.5)), 6),
            round(float(np.percentile(values, 97.5)), 6)]


# ---------------------------------------------------------------------------
# M1-M3: quadrant-level decision metrics
# ---------------------------------------------------------------------------

def _quadrant_point_metrics(q_true: np.ndarray, q_pred: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out['quadrant_exact_match'] = float((q_true == q_pred).mean())
    out['quadrant_kappa'] = float(cohen_kappa_score(q_true, q_pred, labels=list(QUADRANT_IDS)))

    # Q2 treated as the positive class: the paradoxical molecules.
    is_q2_true = (q_true == PARADOX_QUADRANT).astype(int)
    is_q2_pred = (q_pred == PARADOX_QUADRANT).astype(int)
    out['paradox_recall'] = float(recall_score(is_q2_true, is_q2_pred, zero_division=0))
    out['paradox_precision'] = float(precision_score(is_q2_true, is_q2_pred, zero_division=0))
    out['paradox_f1'] = float(f1_score(is_q2_true, is_q2_pred, zero_division=0))

    # Paradox leakage: true-Q2 molecules the model would progress as Q1. The single
    # most consequential error in CNS triage - a compound that cannot reach the brain
    # being advanced as though it can.
    n_q2 = int(is_q2_true.sum())
    leaked = int(((q_true == PARADOX_QUADRANT) & (q_pred == PROGRESS_QUADRANT)).sum())
    out['paradox_leakage_rate'] = float(leaked / n_q2) if n_q2 else float('nan')
    out['paradox_leaked_n'] = float(leaked)
    out['paradox_true_n'] = float(n_q2)
    return out


def quadrant_metrics(
    y_bbb_true: Sequence[int],
    y_pgp_true: Sequence[int],
    y_bbb_pred: Sequence[int],
    y_pgp_pred: Sequence[int],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    M1, M2 and M3 together, with percentile bootstrap 95% intervals.

    Also returns the full 4x4 quadrant confusion matrix (rows = true, cols =
    predicted, ordered Q1..Q4) so the kind of error is visible, not just the rate.
    """
    q_true = assign_quadrants(y_bbb_true, y_pgp_true)
    q_pred = assign_quadrants(y_bbb_pred, y_pgp_pred)
    n = len(q_true)

    if n == 0:
        return {'n': 0, 'note': 'no dual-labelled molecules available'}

    result: Dict[str, Any] = {'n': int(n)}
    result.update({k: (round(v, 6) if v == v else None)
                   for k, v in _quadrant_point_metrics(q_true, q_pred).items()})

    cm = confusion_matrix(q_true, q_pred, labels=list(QUADRANT_IDS))
    result['quadrant_confusion'] = {
        f'true_Q{QUADRANT_IDS[i]}': {f'pred_Q{QUADRANT_IDS[j]}': int(cm[i, j])
                                     for j in range(len(QUADRANT_IDS))}
        for i in range(len(QUADRANT_IDS))
    }
    result['quadrant_support'] = {f'Q{q}': int((q_true == q).sum()) for q in QUADRANT_IDS}
    result['quadrant_names'] = {f'Q{q}': QUADRANT_NAMES[q] for q in QUADRANT_IDS}

    if n_bootstraps and n_bootstraps > 0:
        rng = np.random.RandomState(seed)
        keys = ['quadrant_exact_match', 'quadrant_kappa', 'paradox_recall',
                'paradox_precision', 'paradox_f1', 'paradox_leakage_rate']
        boots: Dict[str, List[float]] = {k: [] for k in keys}
        for _ in range(n_bootstraps):
            idx = rng.randint(0, n, n)
            bt, bp = q_true[idx], q_pred[idx]
            if len(np.unique(bt)) < 2:
                continue
            m = _quadrant_point_metrics(bt, bp)
            for k in keys:
                if m[k] == m[k]:
                    boots[k].append(m[k])
        for k in keys:
            ci = _percentile_ci(boots[k])
            if ci:
                result[f'{k}_CI95'] = ci
        result['n_bootstraps'] = int(n_bootstraps)

    return result


# ---------------------------------------------------------------------------
# M4: efflux-disparity gap
# ---------------------------------------------------------------------------

def efflux_disparity_gap(
    y_bbb_true: Sequence[int],
    y_bbb_prob: Sequence[float],
    y_pgp_true: Sequence[int],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    M4: is the BBB model systematically worse exactly where efflux is present?

        EDG = AUC_BBB(P-gp non-substrates) - AUC_BBB(P-gp substrates)

    A positive gap means permeability prediction degrades on efflux substrates.
    This turns the measured near-independence of the two labels into a score any
    BBB model can be evaluated on.

    Both strata must contain both BBB classes or the AUC is undefined there; the
    substrate stratum is the small one (~64 molecules), so the interval is wide
    and must be reported with the point estimate, never without.
    """
    y = np.asarray(y_bbb_true).astype(int)
    p = np.asarray(y_bbb_prob, dtype=float)
    s = np.asarray(y_pgp_true).astype(int)

    def _auc(mask: np.ndarray) -> Optional[float]:
        if mask.sum() == 0:
            return None
        yy, pp = y[mask], p[mask]
        if len(np.unique(yy)) < 2:
            return None
        return float(roc_auc_score(yy, pp))

    non_sub = s == 0
    sub = s == 1
    auc_non = _auc(non_sub)
    auc_sub = _auc(sub)

    result: Dict[str, Any] = {
        'n_total': int(len(y)),
        'n_non_substrate': int(non_sub.sum()),
        'n_substrate': int(sub.sum()),
        'auc_non_substrate': round(auc_non, 6) if auc_non is not None else None,
        'auc_substrate': round(auc_sub, 6) if auc_sub is not None else None,
    }
    if auc_non is None or auc_sub is None:
        result['note'] = 'one stratum lacks both BBB classes; gap undefined'
        return result

    result['efflux_disparity_gap'] = round(auc_non - auc_sub, 6)

    if n_bootstraps and n_bootstraps > 0:
        rng = np.random.RandomState(seed)
        gaps: List[float] = []
        n = len(y)
        for _ in range(n_bootstraps):
            idx = rng.randint(0, n, n)
            yy, pp, ss = y[idx], p[idx], s[idx]
            m_non, m_sub = ss == 0, ss == 1
            if m_non.sum() == 0 or m_sub.sum() == 0:
                continue
            if len(np.unique(yy[m_non])) < 2 or len(np.unique(yy[m_sub])) < 2:
                continue
            gaps.append(float(roc_auc_score(yy[m_non], pp[m_non])
                              - roc_auc_score(yy[m_sub], pp[m_sub])))
        ci = _percentile_ci(gaps)
        if ci:
            result['efflux_disparity_gap_CI95'] = ci
            # Honest reading: if the interval spans zero, there is no detected gap.
            result['gap_excludes_zero'] = bool(ci[0] > 0 or ci[1] < 0)
        result['n_bootstraps'] = int(n_bootstraps)

    return result


# ---------------------------------------------------------------------------
# M5: expected decision regret
# ---------------------------------------------------------------------------

def decision_regret_curve(
    y_bbb_true: Sequence[int],
    y_pgp_true: Sequence[int],
    y_bbb_pred: Sequence[int],
    y_pgp_pred: Sequence[int],
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> Dict[str, Any]:
    """
    M5: expected cost of the progression decision, as a curve in the cost ratio.

    The decision is binary: progress this molecule as a CNS candidate (predicted Q1)
    or do not. Costs are asymmetric:

        predict Q1, truth is not Q1  ->  cost lambda   (wasted development)
        truth is Q1, predicted not Q1 ->  cost 1       (lost opportunity)
        otherwise                     ->  cost 0

        regret(lambda) = (lambda * n_false_progress + n_missed_opportunity) / n

    lambda = c_FP / v_hit, the cost-ratio parameterisation used by Budget-Sensitive
    Discovery Scoring. Reporting a CURVE rather than a single number avoids inventing
    dollar values, which is what makes the metric defensible.

    At lambda = 1 this reduces exactly to the Q1-versus-rest error rate. As lambda
    grows it is dominated by false progressions, so the per-true-quadrant breakdown
    of those errors is returned alongside: the Q2 row is the paradoxical molecules
    being advanced as though they reach the brain.
    """
    q_true = assign_quadrants(y_bbb_true, y_pgp_true)
    q_pred = assign_quadrants(y_bbb_pred, y_pgp_pred)
    n = len(q_true)
    if n == 0:
        return {'n': 0, 'note': 'no dual-labelled molecules available'}

    progress_pred = q_pred == PROGRESS_QUADRANT
    progress_true = q_true == PROGRESS_QUADRANT

    false_progress = progress_pred & ~progress_true
    missed = progress_true & ~progress_pred

    n_fp = int(false_progress.sum())
    n_fn = int(missed.sum())

    curve = [
        {'lambda': float(lam), 'regret': round(float((lam * n_fp + n_fn) / n), 5)}
        for lam in lambdas
    ]

    # Which true quadrants are being wrongly progressed.
    breakdown = {
        f'Q{q}': int((false_progress & (q_true == q)).sum()) for q in QUADRANT_IDS
    }

    return {
        'n': int(n),
        'n_false_progress': n_fp,
        'n_missed_opportunity': n_fn,
        'false_progress_by_true_quadrant': breakdown,
        'paradoxical_false_progress': breakdown.get(f'Q{PARADOX_QUADRANT}', 0),
        'regret_curve': curve,
        'regret_at_lambda_1': round(float((n_fp + n_fn) / n), 5),
        'q1_vs_rest_error_rate': round(float((n_fp + n_fn) / n), 5),
    }


# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------

def full_decision_report(
    y_bbb_true: Sequence[int],
    y_pgp_true: Sequence[int],
    y_bbb_pred: Sequence[int],
    y_pgp_pred: Sequence[int],
    y_bbb_prob: Optional[Sequence[float]] = None,
    n_bootstraps: int = 1000,
    seed: int = 42,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> Dict[str, Any]:
    """Runs M1-M5 and returns one nested report."""
    report: Dict[str, Any] = {
        'quadrant': quadrant_metrics(
            y_bbb_true, y_pgp_true, y_bbb_pred, y_pgp_pred,
            n_bootstraps=n_bootstraps, seed=seed,
        ),
        'regret': decision_regret_curve(
            y_bbb_true, y_pgp_true, y_bbb_pred, y_pgp_pred, lambdas=lambdas,
        ),
    }
    if y_bbb_prob is not None:
        report['efflux_disparity'] = efflux_disparity_gap(
            y_bbb_true, y_bbb_prob, y_pgp_true, n_bootstraps=n_bootstraps, seed=seed,
        )
    return report


def flatten_decision_report(label: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens a report into one CSV row."""
    q = report.get('quadrant', {})
    r = report.get('regret', {})
    e = report.get('efflux_disparity', {})
    row: Dict[str, Any] = {
        'Evaluation': label,
        'N_dual_labelled': q.get('n'),
        'Quadrant_ExactMatch': q.get('quadrant_exact_match'),
        'Quadrant_ExactMatch_CI95': _fmt_ci(q.get('quadrant_exact_match_CI95')),
        'Quadrant_Kappa': q.get('quadrant_kappa'),
        'Quadrant_Kappa_CI95': _fmt_ci(q.get('quadrant_kappa_CI95')),
        'Paradox_Recall': q.get('paradox_recall'),
        'Paradox_Recall_CI95': _fmt_ci(q.get('paradox_recall_CI95')),
        'Paradox_Precision': q.get('paradox_precision'),
        'Paradox_F1': q.get('paradox_f1'),
        'Paradox_LeakageRate': q.get('paradox_leakage_rate'),
        'Paradox_Leaked_N': q.get('paradox_leaked_n'),
        'Paradox_True_N': q.get('paradox_true_n'),
        'EffluxDisparityGap': e.get('efflux_disparity_gap'),
        'EffluxDisparityGap_CI95': _fmt_ci(e.get('efflux_disparity_gap_CI95')),
        'EDG_ExcludesZero': e.get('gap_excludes_zero'),
        'AUC_NonSubstrate': e.get('auc_non_substrate'),
        'AUC_Substrate': e.get('auc_substrate'),
        'N_FalseProgress': r.get('n_false_progress'),
        'N_MissedOpportunity': r.get('n_missed_opportunity'),
        'ParadoxicalFalseProgress': r.get('paradoxical_false_progress'),
        'Q1vsRest_ErrorRate': r.get('q1_vs_rest_error_rate'),
    }
    for point in r.get('regret_curve', []):
        row[f'Regret_lambda_{point["lambda"]:g}'] = point['regret']
    return row


def _fmt_ci(ci: Optional[List[float]]) -> Optional[str]:
    return f'{ci[0]:.4f} - {ci[1]:.4f}' if ci else None
