"""
Verification Suite for the Decision-Aware Dual-Barrier Metrics.

Checks each metric against cases whose answer is known by construction, before any
of them is computed on real predictions. A metric that is wrong in a way nobody
tested is worse than no metric at all.

Exit code is 0 only if every check passes.
"""

import os
import sys
from typing import List, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.decision_matrix import (
    quadrant_id, determine_dual_barrier_quadrant, BENCHMARK_DRUGS, QUADRANT_IDS,
)
from gnn.decision_metrics import (
    assign_quadrants, quadrant_metrics, efflux_disparity_gap,
    decision_regret_curve, full_decision_report, flatten_decision_report,
)

_results: List[Tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = '') -> None:
    _results.append((name, passed, detail))
    print(f'  [{"PASS" if passed else "FAIL"}] {name}' + (f' - {detail}' if detail else ''))


def section(title: str) -> None:
    print(f'\n{title}')
    print('-' * len(title))


def verify_quadrant_convention() -> None:
    section('1. Quadrant convention is single-sourced')
    ok = True
    for b in (0, 1):
        for p in (0, 1):
            label = determine_dual_barrier_quadrant(b, p)[0]
            if f'Quadrant {quadrant_id(b, p)}:' not in label:
                ok = False
    check('quadrant_id matches determine_dual_barrier_quadrant for all 4 inputs', ok)

    # Derive (bbb, pgp) from each benchmark drug's ground-truth string and confirm
    # quadrant_id reproduces the expected_quadrant recorded alongside it.
    mismatches = []
    for name, d in BENCHMARK_DRUGS.items():
        gt = d['ground_truth']
        bbb = 1 if 'BBB+' in gt else 0
        pgp = 1 if 'P-gp+' in gt else 0
        if quadrant_id(bbb, pgp) != d['expected_quadrant']:
            mismatches.append((name, quadrant_id(bbb, pgp), d['expected_quadrant']))
    check('quadrant_id reproduces expected_quadrant for all BENCHMARK_DRUGS',
          not mismatches,
          f'{len(BENCHMARK_DRUGS)} drugs checked' if not mismatches else str(mismatches))

    q = assign_quadrants([1, 1, 0, 0], [0, 1, 0, 1])
    check('assign_quadrants vectorises correctly', list(q) == [1, 2, 3, 4], str(list(q)))


def verify_perfect_and_degenerate() -> None:
    section('2. Perfect and degenerate predictors')
    rng = np.random.RandomState(0)
    n = 400
    y_bbb = rng.randint(0, 2, n)
    y_pgp = rng.randint(0, 2, n)

    m = quadrant_metrics(y_bbb, y_pgp, y_bbb, y_pgp, n_bootstraps=200, seed=0)
    check('perfect predictor: exact match 1.0', m['quadrant_exact_match'] == 1.0)
    check('perfect predictor: kappa 1.0', m['quadrant_kappa'] == 1.0)
    check('perfect predictor: paradox recall 1.0', m['paradox_recall'] == 1.0)
    check('perfect predictor: paradox leakage 0.0', m['paradox_leakage_rate'] == 0.0)

    r = decision_regret_curve(y_bbb, y_pgp, y_bbb, y_pgp)
    check('perfect predictor: zero regret at every lambda',
          all(pt['regret'] == 0.0 for pt in r['regret_curve']))

    # A single-axis BBB model has no efflux output. Scoring it means assuming
    # P-gp- for everything, which is exactly what a BBB-only pipeline implies.
    single_axis_pgp = np.zeros(n, dtype=int)
    m2 = quadrant_metrics(y_bbb, y_pgp, y_bbb, single_axis_pgp, n_bootstraps=0)
    check('single-axis BBB model: paradox recall exactly 0',
          m2['paradox_recall'] == 0.0, 'cannot express BBB+/P-gp+ by construction')
    check('single-axis BBB model: paradox leakage rate 1.0',
          m2['paradox_leakage_rate'] == 1.0,
          'every true Q2 molecule is progressed as Q1')

    # All-Q2 predictor: recall 1.0, precision = prevalence of Q2.
    q_true = assign_quadrants(y_bbb, y_pgp)
    n_q2 = int((q_true == 2).sum())
    m3 = quadrant_metrics(y_bbb, y_pgp, np.ones(n, dtype=int), np.ones(n, dtype=int),
                          n_bootstraps=0)
    expected_prec = n_q2 / n
    check('all-Q2 predictor: paradox recall 1.0', m3['paradox_recall'] == 1.0)
    check('all-Q2 predictor: paradox precision equals Q2 prevalence',
          abs(m3['paradox_precision'] - expected_prec) < 1e-9,
          f'{m3["paradox_precision"]:.4f} vs {expected_prec:.4f}')


def verify_confusion_and_support() -> None:
    section('3. Confusion matrix and support reconcile')
    rng = np.random.RandomState(1)
    n = 300
    y_bbb, y_pgp = rng.randint(0, 2, n), rng.randint(0, 2, n)
    p_bbb, p_pgp = rng.randint(0, 2, n), rng.randint(0, 2, n)
    m = quadrant_metrics(y_bbb, y_pgp, p_bbb, p_pgp, n_bootstraps=0)

    total = sum(v for row in m['quadrant_confusion'].values() for v in row.values())
    check('confusion matrix sums to n', total == n, f'{total} vs {n}')

    support_total = sum(m['quadrant_support'].values())
    check('quadrant support sums to n', support_total == n, f'{support_total} vs {n}')

    diag = sum(m['quadrant_confusion'][f'true_Q{q}'][f'pred_Q{q}'] for q in QUADRANT_IDS)
    check('diagonal equals exact-match count',
          abs(diag / n - m['quadrant_exact_match']) < 1e-5,
          f'{diag}/{n} = {diag / n:.4f} vs {m["quadrant_exact_match"]:.4f}')


def verify_efflux_disparity() -> None:
    section('4. Efflux-disparity gap (M4)')
    rng = np.random.RandomState(2)
    n = 600
    y_pgp = rng.randint(0, 2, n)
    y_bbb = rng.randint(0, 2, n)

    # Case A: probability is equally informative in both strata -> gap near zero.
    prob = np.where(y_bbb == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    a = efflux_disparity_gap(y_bbb, prob, y_pgp, n_bootstraps=300, seed=2)
    check('equally informative strata: gap is near zero',
          abs(a['efflux_disparity_gap']) < 0.05, f'gap = {a["efflux_disparity_gap"]}')
    check('equally informative strata: CI spans zero',
          not a.get('gap_excludes_zero', True), f'CI {a.get("efflux_disparity_gap_CI95")}')

    # Case B: deliberately destroy the signal on substrates only -> gap must be
    # clearly positive and its interval must exclude zero.
    prob_b = prob.copy()
    sub = y_pgp == 1
    prob_b[sub] = rng.uniform(0.0, 1.0, int(sub.sum()))
    b = efflux_disparity_gap(y_bbb, prob_b, y_pgp, n_bootstraps=300, seed=2)
    check('degraded-on-substrates: gap is clearly positive',
          b['efflux_disparity_gap'] > 0.2, f'gap = {b["efflux_disparity_gap"]}')
    check('degraded-on-substrates: CI excludes zero',
          bool(b.get('gap_excludes_zero')), f'CI {b.get("efflux_disparity_gap_CI95")}')
    check('strata sizes reported',
          b['n_non_substrate'] + b['n_substrate'] == n,
          f'{b["n_non_substrate"]} non-substrate + {b["n_substrate"]} substrate')

    # Single-class stratum must be refused, not silently scored.
    y_one = np.ones(n, dtype=int)
    c = efflux_disparity_gap(y_one, prob, y_pgp, n_bootstraps=0)
    check('single-class stratum returns a note instead of a number',
          'efflux_disparity_gap' not in c and 'note' in c, c.get('note', ''))


def verify_regret() -> None:
    section('5. Decision regret curve (M5)')
    rng = np.random.RandomState(3)
    n = 500
    y_bbb, y_pgp = rng.randint(0, 2, n), rng.randint(0, 2, n)
    p_bbb, p_pgp = rng.randint(0, 2, n), rng.randint(0, 2, n)
    r = decision_regret_curve(y_bbb, y_pgp, p_bbb, p_pgp)

    check('regret at lambda=1 equals the Q1-vs-rest error rate',
          abs(r['regret_at_lambda_1'] - r['q1_vs_rest_error_rate']) < 1e-12,
          f'{r["regret_at_lambda_1"]}')

    manual = (r['n_false_progress'] + r['n_missed_opportunity']) / n
    check('lambda=1 regret matches a hand calculation',
          abs(r['regret_at_lambda_1'] - manual) < 1e-12, f'{manual:.5f}')

    regrets = [pt['regret'] for pt in r['regret_curve']]
    check('regret is non-decreasing in lambda',
          all(b >= a - 1e-12 for a, b in zip(regrets, regrets[1:])),
          f'{regrets[0]:.4f} .. {regrets[-1]:.4f}')

    fp_sum = sum(r['false_progress_by_true_quadrant'].values())
    check('false-progress breakdown sums to the false-progress count',
          fp_sum == r['n_false_progress'], f'{fp_sum} vs {r["n_false_progress"]}')
    check('paradoxical false progress is the Q2 row of that breakdown',
          r['paradoxical_false_progress'] == r['false_progress_by_true_quadrant']['Q2'],
          f'{r["paradoxical_false_progress"]} molecules')

    # As lambda grows, false progressions must dominate.
    big = decision_regret_curve(y_bbb, y_pgp, p_bbb, p_pgp, lambdas=[1e6])
    dominated = abs(big['regret_curve'][0]['regret']
                    - 1e6 * big['n_false_progress'] / n) / (1e6 * max(big['n_false_progress'], 1) / n)
    check('large lambda is dominated by false progressions', dominated < 1e-3,
          f'relative residual {dominated:.2e}')


def verify_report_shape() -> None:
    section('6. Report assembly and flattening')
    rng = np.random.RandomState(4)
    n = 216  # the real dual-labelled size
    y_bbb, y_pgp = rng.randint(0, 2, n), rng.randint(0, 2, n)
    p_bbb, p_pgp = rng.randint(0, 2, n), rng.randint(0, 2, n)
    prob = rng.uniform(0, 1, n)

    rep = full_decision_report(y_bbb, y_pgp, p_bbb, p_pgp, y_bbb_prob=prob,
                              n_bootstraps=200, seed=4)
    check('report contains all three blocks',
          set(rep.keys()) == {'quadrant', 'regret', 'efflux_disparity'},
          str(sorted(rep.keys())))

    row = flatten_decision_report('synthetic', rep)
    required = ['Quadrant_ExactMatch', 'Quadrant_Kappa', 'Paradox_Recall',
                'Paradox_LeakageRate', 'EffluxDisparityGap', 'Q1vsRest_ErrorRate']
    missing = [k for k in required if k not in row]
    check('flattened row has every headline field', not missing, str(missing))
    check('flattened row carries the regret curve',
          any(k.startswith('Regret_lambda_') for k in row),
          f'{sum(1 for k in row if k.startswith("Regret_lambda_"))} lambda columns')

    empty = quadrant_metrics([], [], [], [], n_bootstraps=0)
    check('empty input returns a note rather than raising',
          empty.get('n') == 0 and 'note' in empty, empty.get('note', ''))


def main() -> int:
    print('=' * 78)
    print('DECISION-AWARE METRIC VERIFICATION SUITE')
    print('=' * 78)

    verify_quadrant_convention()
    verify_perfect_and_degenerate()
    verify_confusion_and_support()
    verify_efflux_disparity()
    verify_regret()
    verify_report_shape()

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = [n for n, ok, _ in _results if not ok]
    print('\n' + '=' * 78)
    print(f'RESULT: {passed}/{len(_results)} checks passed')
    if failed:
        print('FAILED:')
        for name in failed:
            print(f'  - {name}')
    print('=' * 78)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
