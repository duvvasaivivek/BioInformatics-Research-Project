"""
Dual-Barrier Corpus Assembly for the Multi-Task GNN.

Merges the B3DB blood-brain barrier benchmark and the Broccatelli P-glycoprotein
substrate benchmark into one molecule-level corpus in which each molecule carries
a BBB label, a P-gp label, or both, together with a per-task validity mask.

Why masks are unavoidable: only ~216 molecules carry both labels, so roughly 97.6%
of the merged corpus has exactly one of the two. A plain summed cross-entropy over
two heads is therefore invalid - see gnn/losses.py.

The join key is the full InChIKey. Neither canonical-SMILES nor InChIKey generation
existed anywhere in this repository before this module; core/features.py is entirely
descriptor- and fingerprint-oriented and offers nothing reusable for structure matching.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import inchi

RDLogger.DisableLog('rdApp.*')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

B3DB_PATH = os.path.join(DATA_DIR, 'B3DB_classification.tsv')
PGP_PATH = os.path.join(DATA_DIR, 'pgp_broccatelli.tab')

TASKS = ('bbb', 'pgp')

# Semantics, carried forward from the existing preprocessing scripts so the GNN
# arm and the XGBoost arm cannot disagree about what a label means:
#   BBB  1 = BBB+ (permeant)       - BBB/bbb_data_preprocessing.py:46
#   P-gp 1 = substrate (effluxed)  - Pgp/pgp_data_preprocessing.py (TDC Y passthrough)
BBB_LABEL_MAP = {'BBB+': 1, 'BBB-': 0}


def canonical_identity(smiles: str) -> Optional[Tuple[str, str]]:
    """
    Returns (canonical_smiles, inchikey) for a SMILES string, or None if invalid.

    The InChIKey is the join key: it is stereo-sensitive and invariant to the
    SMILES spelling differences between the two source benchmarks.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        key = inchi.MolToInchiKey(mol)
    except Exception:
        return None
    if not key:
        return None
    return Chem.MolToSmiles(mol), key


def _load_source(
    path: str,
    smiles_col: str,
    label_col: str,
    label_map: Optional[Dict[Any, int]],
    task: str,
    extra_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads one benchmark, canonicalises it, and collapses duplicate structures."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Required dataset missing: {path}')

    df = pd.read_csv(path, sep='\t')
    raw_rows = len(df)

    records = []
    invalid = 0
    for _, row in df.iterrows():
        ident = canonical_identity(row[smiles_col])
        if ident is None:
            invalid += 1
            continue
        canon, key = ident

        raw_label = row[label_col]
        if pd.isna(raw_label):
            invalid += 1
            continue
        if label_map is not None:
            if raw_label not in label_map:
                invalid += 1
                continue
            label = label_map[raw_label]
        else:
            label = int(raw_label)

        rec = {'inchikey': key, 'canonical_smiles': canon, f'y_{task}': label}
        for col in (extra_cols or []):
            rec[col] = row.get(col, np.nan)
        records.append(rec)

    out = pd.DataFrame(records)

    # Collapse duplicate structures. Conflicting labels on the same structure are
    # dropped rather than silently resolved by row order - the reference pipeline's
    # drop_duplicates(keep='first') silently lets whichever source came first win.
    conflicts = 0
    if len(out):
        grp = out.groupby('inchikey')[f'y_{task}'].nunique()
        conflicting_keys = set(grp[grp > 1].index)
        conflicts = len(conflicting_keys)
        if conflicting_keys:
            out = out[~out['inchikey'].isin(conflicting_keys)]
        out = out.drop_duplicates(subset=['inchikey'], keep='first').reset_index(drop=True)

    stats = {
        'path': os.path.relpath(path, BASE_DIR),
        'raw_rows': raw_rows,
        'invalid_or_unlabelled': invalid,
        'conflicting_structures_dropped': conflicts,
        'unique_structures': len(out),
        'positive': int(out[f'y_{task}'].sum()) if len(out) else 0,
        'negative': int((1 - out[f'y_{task}']).sum()) if len(out) else 0,
    }
    return out, stats


def build_corpus(
    b3db_path: str = B3DB_PATH,
    pgp_path: str = PGP_PATH,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds the merged dual-barrier corpus.

    Returns (corpus, stats) where corpus has one row per unique structure with
    columns: inchikey, canonical_smiles, y_bbb, m_bbb, y_pgp, m_pgp, logBB.

    y_* is 0/1 and is meaningless wherever the matching m_* mask is 0.
    """
    bbb_df, bbb_stats = _load_source(
        b3db_path, smiles_col='SMILES', label_col='BBB+/BBB-',
        label_map=BBB_LABEL_MAP, task='bbb', extra_cols=['logBB', 'compound_name'],
    )
    pgp_df, pgp_stats = _load_source(
        pgp_path, smiles_col='Drug', label_col='Y',
        label_map=None, task='pgp', extra_cols=['Drug_ID'],
    )

    merged = pd.merge(
        bbb_df, pgp_df[['inchikey', 'y_pgp']],
        on='inchikey', how='outer', suffixes=('', '_pgp_src'),
    )

    # An outer join leaves canonical_smiles empty for P-gp-only molecules.
    pgp_smiles = pgp_df.set_index('inchikey')['canonical_smiles']
    missing = merged['canonical_smiles'].isna()
    merged.loc[missing, 'canonical_smiles'] = merged.loc[missing, 'inchikey'].map(pgp_smiles)

    merged['m_bbb'] = merged['y_bbb'].notna().astype(int)
    merged['m_pgp'] = merged['y_pgp'].notna().astype(int)
    merged['y_bbb'] = merged['y_bbb'].fillna(0).astype(int)
    merged['y_pgp'] = merged['y_pgp'].fillna(0).astype(int)
    if 'logBB' not in merged.columns:
        merged['logBB'] = np.nan
    merged['logBB'] = pd.to_numeric(merged['logBB'], errors='coerce')

    merged = merged.dropna(subset=['canonical_smiles']).reset_index(drop=True)

    both = merged[(merged['m_bbb'] == 1) & (merged['m_pgp'] == 1)]
    contingency = {
        'bbb_pos_pgp_pos': int(((both['y_bbb'] == 1) & (both['y_pgp'] == 1)).sum()),
        'bbb_pos_pgp_neg': int(((both['y_bbb'] == 1) & (both['y_pgp'] == 0)).sum()),
        'bbb_neg_pgp_pos': int(((both['y_bbb'] == 0) & (both['y_pgp'] == 1)).sum()),
        'bbb_neg_pgp_neg': int(((both['y_bbb'] == 0) & (both['y_pgp'] == 0)).sum()),
    }

    stats: Dict[str, Any] = {
        'bbb_source': bbb_stats,
        'pgp_source': pgp_stats,
        'corpus_size': len(merged),
        'bbb_labelled': int(merged['m_bbb'].sum()),
        'pgp_labelled': int(merged['m_pgp'].sum()),
        'dual_labelled': int(len(both)),
        'bbb_only': int(((merged['m_bbb'] == 1) & (merged['m_pgp'] == 0)).sum()),
        'pgp_only': int(((merged['m_bbb'] == 0) & (merged['m_pgp'] == 1)).sum()),
        'single_label_fraction': round(
            float(((merged['m_bbb'] + merged['m_pgp']) == 1).sum()) / max(len(merged), 1), 4
        ),
        'logbb_available': int(merged['logBB'].notna().sum()),
        'dual_label_contingency': contingency,
    }

    if verbose:
        print(f"[DATA] B3DB      : {bbb_stats['raw_rows']} rows -> "
              f"{bbb_stats['unique_structures']} structures "
              f"({bbb_stats['positive']} BBB+, {bbb_stats['negative']} BBB-)")
        print(f"[DATA] Broccatelli: {pgp_stats['raw_rows']} rows -> "
              f"{pgp_stats['unique_structures']} structures "
              f"({pgp_stats['positive']} substrate, {pgp_stats['negative']} non-substrate)")
        print(f"[DATA] Merged corpus : {stats['corpus_size']} unique structures")
        print(f"[DATA]   BBB-labelled : {stats['bbb_labelled']}")
        print(f"[DATA]   Pgp-labelled : {stats['pgp_labelled']}")
        print(f"[DATA]   Dual-labelled: {stats['dual_labelled']}  "
              f"({100 * stats['single_label_fraction']:.1f}% of corpus has one label only)")
        print(f"[DATA]   logBB values : {stats['logbb_available']}")

    return merged, stats


def label_independence(corpus: pd.DataFrame) -> Dict[str, Any]:
    """
    Tests association between the two labels on the dual-labelled subset.

    This is the empirical justification for pairing these two tasks: near-independent
    labels carry complementary information. Reported as an association test with its
    limits, not as proof of independence - at this sample size the odds-ratio
    interval still admits a moderate association.
    """
    from scipy import stats as sps

    both = corpus[(corpus['m_bbb'] == 1) & (corpus['m_pgp'] == 1)]
    a = int(((both['y_bbb'] == 1) & (both['y_pgp'] == 1)).sum())
    b = int(((both['y_bbb'] == 1) & (both['y_pgp'] == 0)).sum())
    c = int(((both['y_bbb'] == 0) & (both['y_pgp'] == 1)).sum())
    d = int(((both['y_bbb'] == 0) & (both['y_pgp'] == 0)).sum())
    n = a + b + c + d

    table = np.array([[a, b], [c, d]])
    result: Dict[str, Any] = {
        'n': n,
        'table': {'bbb+_pgp+': a, 'bbb+_pgp-': b, 'bbb-_pgp+': c, 'bbb-_pgp-': d},
    }
    if min(a, b, c, d) == 0 or n == 0:
        result['note'] = 'degenerate table; association statistics not computed'
        return result

    chi2, p_chi2, _, _ = sps.chi2_contingency(table)
    odds, p_fisher = sps.fisher_exact(table)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    result.update({
        'chi2': round(float(chi2), 4),
        'p_chi2': round(float(p_chi2), 4),
        'odds_ratio': round(float(odds), 4),
        'odds_ratio_ci95': [round(float(np.exp(np.log(odds) - 1.96 * se)), 4),
                            round(float(np.exp(np.log(odds) + 1.96 * se)), 4)],
        'cramers_v': round(float(np.sqrt(chi2 / n)), 4),
        'p_pgp_given_bbb_pos': round(a / (a + b), 4),
        'p_pgp_given_bbb_neg': round(c / (c + d), 4),
        'interpretation': (
            'BBB label is not a usable proxy for efflux status; '
            'any residual association is bounded as weak'
        ),
    })
    return result
