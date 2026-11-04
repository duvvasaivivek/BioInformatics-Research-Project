"""Refresh marked result tables in the two public documents, or check for drift."""
import argparse
import csv
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / 'results'
DOCS = (BASE.parent / 'README.md', BASE.parent / 'RESEARCH_PAPER_MANUSCRIPT.md')


def rows(name):
    with (RESULTS / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in records])


def build_tables():
    tables = {}
    columns = ['BBB_Random', 'BBB_Scaffold', 'Pgp_Random', 'Pgp_Scaffold']
    records = []
    for row in rows('validation_metrics_summary.csv'):
        records.append([row['Metric']] + [
            f"{float(row[c + '_Estimate']):.4f} {row[c + '_95CI']}" for c in columns])
    tables['validation'] = table(['Metric', 'BBB random', 'BBB scaffold', 'P-gp random', 'P-gp scaffold'], records)
    tables['ablation'] = table(['Task', 'Configuration', 'Features', 'Accuracy', 'F1', 'ROC-AUC'], [
        [r['Task'], r['Configuration'], r['Num_Features'], r['Accuracy'], r['F1_Score'], r['ROC_AUC']]
        for r in rows('ablation_study_results.csv')])
    tables['shap'] = table(['Descriptor', 'BBB mean absolute SHAP', 'P-gp mean absolute SHAP',
                            'BBB mean signed SHAP', 'P-gp mean signed SHAP'], [
        [r['Property']] + [f'{float(r[c]):.4f}' for c in
         ['BBB_Mean_Abs_SHAP','Pgp_Mean_Abs_SHAP','BBB_Signed_SHAP','Pgp_Signed_SHAP']]
        for r in rows('shared_properties_quantitative_comparison.csv')])
    tables['gnn'] = table(['Task', 'Split', 'Single-task AUC', 'Multitask AUC', 'Delta',
                           'Paired delta 95% CI', 'Paired result'], [
        [r['task'].upper(), r['split'], r['single_ROC_AUC'], r['multi_ROC_AUC'],
         f"{float(r['paired_delta_ROC_AUC']):+.4f}",
         f"[{float(r['paired_delta_CI95_low']):+.4f}, {float(r['paired_delta_CI95_high']):+.4f}]",r['paired_verdict']]
        for r in rows('gnn_multitask_delta.csv')])
    tables['seed_variance'] = table(['Task', 'Metric', 'Seeds', 'Mean delta', 'Delta 95% CI', 'MDE', 'Paired p'], [
        [r['task'].upper(),r['metric'],r['n_seeds'],f"{float(r['mean_delta']):+.6f}",
         f"[{float(r['delta_CI95_low']):+.6f}, {float(r['delta_CI95_high']):+.6f}]",r['mde'],r['paired_p']]
        for r in rows('gnn_seedvar_random_contrasts.csv')])
    tables['decisions'] = table(['Strategy', 'N', 'Quadrant accuracy', 'Accuracy 95% CI',
                                 'Kappa', 'Paradox recall', 'Paradox leakage', 'Leaked / true Q2', 'Regret at lambda=1'], [
        [r['Evaluation'],r['N_dual_labelled'],f"{float(r['Quadrant_ExactMatch']):.4f}",
         r['Quadrant_ExactMatch_CI95'],f"{float(r['Quadrant_Kappa']):.4f}",
         f"{float(r['Paradox_Recall']):.4f}",f"{float(r['Paradox_LeakageRate']):.4f}",
         f"{int(float(r['Paradox_Leaked_N']))} / {int(float(r['Paradox_True_N']))}",r['Regret_lambda_1']]
        for r in rows('gnn_decision_metrics.csv')])
    return tables


def block(name, contents):
    return f'<!-- results:{name} -->\n{contents}\n<!-- /results:{name} -->'


def check_tables():
    failures = []
    for path in DOCS:
        text = path.read_text(encoding='utf-8')
        for name, contents in build_tables().items():
            if block(name, contents) not in text:
                failures.append(f'{path.name}: {name}')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not args.check:
        for path in DOCS:
            text = path.read_text(encoding='utf-8')
            for name, contents in build_tables().items():
                pattern = rf'<!-- results:{name} -->.*?<!-- /results:{name} -->'
                text, count = re.subn(pattern, lambda _: block(name, contents), text, flags=re.S)
                if count != 1:
                    raise ValueError(f'Expected one {name} block in {path}')
            path.write_text(text, encoding='utf-8')
    failures = check_tables()
    print('\n'.join(failures) if failures else 'All 12 result tables match their source CSVs.')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
