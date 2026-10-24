"""
Documentation Consistency Checker.

Asserts that the numbers quoted in the project's prose match the artifacts on
disk. Three defects in the manuscript existed purely because figures were copied
from an earlier run's write-up instead of re-read from the results, and the
parameter census in ARCHITECTURE_DEEP_DIVE.md disagreed with its own itemisation.

This script makes that class of error fail loudly instead of surviving into a
submission. Run it after any re-run that changes a saved metric.

Exit code is 0 only if every documented value matches its source.

Usage
-----
    python scripts/verify_doc_consistency.py
"""

import os
import re
import sys
from typing import List, Tuple

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

_results: List[Tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = '') -> None:
    _results.append((name, passed, detail))
    print(f'  [{"PASS" if passed else "FAIL"}] {name}' + (f' - {detail}' if detail else ''))


def read_doc(*parts: str) -> str:
    path = os.path.join(REPO_DIR, *parts)
    if not os.path.exists(path):
        return ''
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def section(title: str) -> None:
    print(f'\n{title}')
    print('-' * len(title))


# ---------------------------------------------------------------------------

def verify_validation_metrics() -> None:
    section('1. Quoted metrics match results/validation_metrics_summary.csv')
    csv = os.path.join(RESULTS_DIR, 'validation_metrics_summary.csv')
    if not os.path.exists(csv):
        check('validation_metrics_summary.csv present', False, 'missing')
        return
    df = pd.read_csv(csv).set_index('Metric')

    def val(metric: str, col: str) -> float:
        return round(float(df.loc[metric, col]), 4)

    authoritative = {
        'BBB ROC-AUC': val('ROC_AUC', 'BBB_Random_Estimate'),
        'BBB Accuracy': val('Accuracy', 'BBB_Random_Estimate'),
        'BBB PR-AUC': val('PR_AUC', 'BBB_Random_Estimate'),
        'P-gp ROC-AUC': val('ROC_AUC', 'Pgp_Random_Estimate'),
    }
    for k, v in authoritative.items():
        print(f'      {k}: {v}')

    # Any doc quoting a *superseded* value is a failure. These are the values
    # that were previously wrong, so their continued absence is the invariant.
    # Bare substring matching, so only values that cannot collide with a
    # legitimate number elsewhere belong here. 0.7762 is deliberately excluded:
    # it is also Levetiracetam's per-molecule P-gp SHAP value in the manuscript
    # case studies. The superseded MCC is checked in table context below instead.
    superseded = ['0.9544', '0.8962', '0.9593', '0.9715', '0.9336']
    docs = {
        'README.md': read_doc('README.md'),
        'RESEARCH_PAPER_MANUSCRIPT.md': read_doc('RESEARCH_PAPER_MANUSCRIPT.md'),
        'ARCHITECTURE_DEEP_DIVE.md': read_doc('ARCHITECTURE_DEEP_DIVE.md'),
        'ARCHITECTURE_DEEP_DIVE.tex': read_doc('ARCHITECTURE_DEEP_DIVE.tex'),
        'docs/SCIENTIFIC_DOCUMENTATION.md': read_doc('B3DB Based Model', 'docs',
                                                     'SCIENTIFIC_DOCUMENTATION.md'),
    }
    # Audit documents legitimately quote the old values while describing the defect.
    AUDIT_DOCS = ('NOVELTY_ASSESSMENT.md', 'MULTITASK_GNN_RUN_LOG.md',
                  'PUBLICATION_STRATEGY.md', 'ARCHITECTURE_FLOW_BLOCK_DIAGRAM.md')
    for name, text in docs.items():
        if not text:
            continue
        hits = [s for s in superseded if s in text]
        check(f'{name} quotes no superseded metric', not hits,
              f'found {hits}' if hits else '')

    # The authoritative values should actually appear where results are reported.
    man = docs['RESEARCH_PAPER_MANUSCRIPT.md']
    if man:
        check('manuscript abstract quotes the current BBB ROC-AUC',
              str(authoritative['BBB ROC-AUC']) in man, str(authoritative['BBB ROC-AUC']))
        check('manuscript abstract quotes the current BBB accuracy',
              str(authoritative['BBB Accuracy']) in man, str(authoritative['BBB Accuracy']))
    print(f'      (audit docs exempt by design: {", ".join(AUDIT_DOCS)})')


def verify_shap_table() -> None:
    section('2. Quoted SHAP attributions match the saved artifact')
    csv = os.path.join(RESULTS_DIR, 'shared_properties_quantitative_comparison.csv')
    if not os.path.exists(csv):
        check('shared_properties CSV present', False, 'missing')
        return
    df = pd.read_csv(csv).set_index('Property')

    man = read_doc('RESEARCH_PAPER_MANUSCRIPT.md')
    sci = read_doc('B3DB Based Model', 'docs', 'SCIENTIFIC_DOCUMENTATION.md')

    missing: List[str] = []
    for prop in df.index:
        v = f"{round(float(df.loc[prop, 'BBB_Mean_Abs_SHAP']), 4):.4f}"
        if man and v not in man:
            missing.append(f'manuscript:{prop}={v}')
        if sci and v not in sci:
            missing.append(f'scidoc:{prop}={v}')
    check('every BBB mean |SHAP| value appears in the prose', not missing,
          f'missing {missing}' if missing else f'{len(df)} properties checked')

    # These numbers are produced by a pipeline that omits the scaler, so any doc
    # presenting them must carry the caveat rather than stating them bare.
    for name, text in (('manuscript', man), ('scidoc', sci)):
        if not text:
            continue
        has = 'Pending recomputation' in text or 'pending recomputation' in text
        check(f'{name} flags the SHAP values as provisional', has,
              'required while run_explainability_analysis.py applies no scaler')


def verify_parameter_census() -> None:
    section('3. GNN parameter census is self-consistent')
    text = read_doc('ARCHITECTURE_DEEP_DIVE.md')
    if not text:
        check('ARCHITECTURE_DEEP_DIVE.md present', False, 'missing')
        return

    VERIFIED_TOTAL = 345514
    m = re.search(r'Complete Parameter Census \(([\d,]+) Total Parameters\)', text)
    heading = int(m.group(1).replace(',', '')) if m else None
    m2 = re.search(r'\*\*Total:\*\* \*\*\$([\d,]+)\$ trainable parameters', text)
    stated = int(m2.group(1).replace(',', '')) if m2 else None

    check('census heading matches the verified count', heading == VERIFIED_TOTAL,
          f'{heading} vs {VERIFIED_TOTAL}')
    check('census total matches the verified count', stated == VERIFIED_TOTAL,
          f'{stated} vs {VERIFIED_TOTAL}')

    # The itemised lines must also add up to the stated total.
    # Exactly one match per census line. The earlier version matched the
    # BatchNorm line with both the generic "= N params" pattern and an explicit
    # BN pattern, double-counting 1,536 and inflating the sum by that amount.
    census = text.split('Complete Parameter Census')[1].split('###')[0]
    items = []
    for line in census.split('\n'):
        if 'params' not in line or not line.strip().startswith('-'):
            continue
        # Sum every value on the line: the classifier-head line carries two
        # ("Linear(256->32) => 8,224 params; Linear(32->2) => 66 params").
        # The pattern only matches a number immediately preceding "params", so
        # the operands of "72 + 2,048 + 2,048 = 4,168 params" are not counted.
        for num in re.findall(r'([\d,]+)\$? params', line):
            items.append(int(num.replace(',', '')))
    total = sum(items)
    check('itemised lines sum to the stated total', total == VERIFIED_TOTAL,
          f'items sum to {total:,} vs {VERIFIED_TOTAL:,} '
          f'({len(items)} line items found)')


def verify_dataset_counts() -> None:
    section('4. Quoted dataset counts match the data')
    pgp = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed_pgp_data.csv'),
                      usecols=['Label'])
    n_pos, n_neg = int((pgp.Label == 1).sum()), int((pgp.Label == 0).sum())
    print(f'      P-gp: {n_pos} substrates / {n_neg} non-substrates ({len(pgp)} total)')

    man = read_doc('RESEARCH_PAPER_MANUSCRIPT.md')
    if man:
        check('manuscript quotes the measured P-gp class balance',
              f'$N={n_pos}$ substrates' in man and f'$N={n_neg}$ non-substrates' in man,
              f'expected {n_pos}/{n_neg}')
        check('manuscript no longer quotes the superseded 668/537 balance',
              '$N=668$ substrates' not in man)

    legacy = os.path.join(REPO_DIR, 'Initial Model', 'Dataset', 'cleaned_bbb_dataset.csv')
    if os.path.exists(legacy):
        src = pd.read_csv(legacy, usecols=['Source']).Source.value_counts().to_dict()
        print(f'      legacy corpus: {src}')
        prov = read_doc('DATASET_AND_FILE_PROVENANCE_CATALOG.md')
        if prov:
            want = f"Adenot ({src.get('Adenot', 0):,}) & Li ({src.get('Li', 0):,})"
            check('provenance catalog quotes the measured Adenot/Li split',
                  want in prov, want)


def verify_docking_count() -> None:
    section('5. Quoted docking compound count matches the results')
    csv = os.path.join(RESULTS_DIR, 'cryoem_pgp_docking_validation.csv')
    if not os.path.exists(csv):
        check('docking CSV present', False, 'missing')
        return
    n = len(pd.read_csv(csv))
    print(f'      docked compounds in results: {n}')

    man = read_doc('RESEARCH_PAPER_MANUSCRIPT.md')
    if man:
        check('manuscript does not claim twenty-two docked compounds',
              'Twenty-two representative therapeutics spanning' not in man)
        check('manuscript states the correct docked count',
              'Twenty-one representative therapeutics' in man, f'n = {n}')
        check('manuscript documents the excluded compound',
              'Topotecan' in man and 'excluded' in man,
              'Topotecan failed kekulisation and produced no pose')


def verify_deleted_files_absent() -> None:
    section('6. No doc presents a deleted file as live')
    gone = ['bbb_model_evaluation.py', 'pgp_model_evaluation.py']
    for rel in ('DATASET_AND_FILE_PROVENANCE_CATALOG.md', 'README.md'):
        text = read_doc(rel)
        if not text:
            continue
        hits = [g for g in gone if g in text]
        check(f'{rel} does not list the deleted evaluation scripts', not hits,
              f'found {hits}' if hits else '')


def main() -> int:
    print('=' * 78)
    print('DOCUMENTATION CONSISTENCY CHECKER')
    print('=' * 78)

    verify_validation_metrics()
    verify_shap_table()
    verify_parameter_census()
    verify_dataset_counts()
    verify_docking_count()
    verify_deleted_files_absent()

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = [n for n, ok, _ in _results if not ok]
    print('\n' + '=' * 78)
    print(f'RESULT: {passed}/{len(_results)} checks passed')
    if failed:
        print('FAILED:')
        for n in failed:
            print(f'  - {n}')
    print('=' * 78)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
