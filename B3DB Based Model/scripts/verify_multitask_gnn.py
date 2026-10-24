"""
Verification Suite for the Multi-Task GNN.

These checks guard the invariants that would silently invalidate every result if
broken. Run this before any training run and after any refactor.

  1. CUDA availability and device plumbing.
  2. Featurizer self-consistency: declared dims match produced tensors.
  3. Legacy-mode parity against the base-paper reference graphs on disk, so the
     refactor and the feature change remain separable.
  4. Corpus assembly reconciles (totals, dual-labelled count, contingency table).
  5. Split integrity for BOTH protocols: no molecule leakage, full coverage,
     dual-labelled molecules keep both labels in one partition, zero scaffold
     overlap for the scaffold split.
  6. Masked loss: a head with no labelled molecules in the batch receives exactly
     zero gradient.
  7. Task weight 0 fully removes a task from the objective.
  8. Model forward produces one logit tensor per task with the right shape.

Exit code is 0 only if every check passes.
"""

import os
import sys
import glob
from typing import List, Tuple

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from gnn.device import get_device, describe_device
from gnn.featurizer import feature_dims, smiles_to_data, FEATURE_MODES
from gnn.data_assembly import build_corpus, label_independence
from gnn.dataset import build_graph_dataset, subset_by_corpus_index, make_loader
from gnn.splits import make_splits, assert_split_integrity
from gnn.model import MultiTaskGNN
from gnn.losses import MaskedMultiTaskLoss, compute_class_weights

REFERENCE_GRAPHS = os.path.join(
    os.path.dirname(BASE_DIR), 'Initial Model', 'graphs'
)

_results: List[Tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = '') -> None:
    _results.append((name, passed, detail))
    mark = 'PASS' if passed else 'FAIL'
    print(f'  [{mark}] {name}' + (f' - {detail}' if detail else ''))


def section(title: str) -> None:
    print(f'\n{title}')
    print('-' * len(title))


def verify_device() -> torch.device:
    section('1. Device and CUDA')
    device = get_device()
    info = describe_device(device)
    for k, v in info.items():
        print(f'      {k}: {v}')
    check('CUDA is available', bool(info['cuda_available']),
          'running on CPU' if not info['cuda_available'] else info.get('gpu_name', ''))
    check('selected device is CUDA', device.type == 'cuda', str(device))
    # Prove a real tensor op lands on the GPU.
    t = torch.ones(8, 8, device=device)
    check('tensor op executes on device', (t @ t).sum().item() == 512.0, str(device))
    return device


def verify_featurizer() -> None:
    section('2. Featurizer self-consistency')
    for mode in FEATURE_MODES:
        atom_dim, bond_dim = feature_dims(mode)
        d = smiles_to_data('CC(=O)Oc1ccccc1C(=O)O', mode=mode)
        ok = d.x.shape[1] == atom_dim and d.edge_attr.shape[1] == bond_dim
        check(f'{mode}: declared dims match tensors',
              ok, f'atom {d.x.shape[1]}=={atom_dim}, bond {d.edge_attr.shape[1]}=={bond_dim}')
    check('legacy mode is the base paper 9/7 encoding', feature_dims('legacy') == (9, 7),
          str(feature_dims('legacy')))
    # Edge list must be symmetric: every bond contributes both directions.
    d = smiles_to_data('c1ccccc1', mode='encoded')
    ei = d.edge_index.numpy()
    pairs = {tuple(p) for p in ei.T.tolist()}
    check('edge list is bidirectional', all((j, i) in pairs for i, j in pairs),
          f'{ei.shape[1]} directed edges')
    check('single-atom molecule yields empty edge tensors',
          smiles_to_data('[Xe]', mode='encoded').edge_index.shape == (2, 0))
    check('invalid SMILES returns None', smiles_to_data('nonsense_xyz') is None)


def verify_legacy_parity() -> None:
    section('3. Legacy parity against base-paper reference graphs')
    files = sorted(glob.glob(os.path.join(REFERENCE_GRAPHS, '*.pt')))
    if not files:
        check('reference graphs present', False,
              f'none found at {REFERENCE_GRAPHS} (they are gitignored; skipping parity)')
        return

    compared = 0
    mismatches = []
    for path in files[:50]:
        ref = torch.load(path, weights_only=False)
        smi = ref.get('smiles')
        if not smi:
            continue
        ours = smiles_to_data(smi, mode='legacy')
        if ours is None:
            mismatches.append((os.path.basename(path), 'ours=None'))
            continue
        if not torch.allclose(ours.x, ref['x'].float()):
            mismatches.append((os.path.basename(path), 'x differs'))
        elif not torch.equal(ours.edge_index, ref['edge_index']):
            mismatches.append((os.path.basename(path), 'edge_index differs'))
        elif not torch.allclose(ours.edge_attr, ref['edge_attr'].float()):
            mismatches.append((os.path.basename(path), 'edge_attr differs'))
        compared += 1

    check('legacy featurizer reproduces reference graphs bit-for-bit',
          not mismatches, f'{compared} graphs compared, {len(mismatches)} mismatches'
          + (f' e.g. {mismatches[:2]}' if mismatches else ''))


def verify_corpus():
    section('4. Corpus assembly')
    corpus, stats = build_corpus(verbose=False)
    check('corpus is non-empty', len(corpus) > 0, f'{len(corpus)} molecules')
    check('every molecule carries at least one label',
          bool(((corpus['m_bbb'] + corpus['m_pgp']) >= 1).all()))
    check('InChIKeys are unique', corpus['inchikey'].is_unique,
          f'{corpus["inchikey"].nunique()} unique of {len(corpus)}')

    total = stats['bbb_labelled'] + stats['pgp_labelled'] - stats['dual_labelled']
    check('label counts reconcile with corpus size', total == stats['corpus_size'],
          f'{stats["bbb_labelled"]} + {stats["pgp_labelled"]} - {stats["dual_labelled"]} '
          f'= {total} vs {stats["corpus_size"]}')

    c = stats['dual_label_contingency']
    s = sum(c.values())
    check('contingency table sums to the dual-labelled count', s == stats['dual_labelled'],
          f'{c} sums to {s}')

    ind = label_independence(corpus)
    print(f'      dual-labelled n={ind["n"]}, Cramers V={ind.get("cramers_v")}, '
          f'OR={ind.get("odds_ratio")} CI{ind.get("odds_ratio_ci95")}, p={ind.get("p_chi2")}')
    check('independence statistics computed', 'cramers_v' in ind)
    return corpus


def verify_splits(corpus) -> None:
    section('5. Split integrity (both protocols)')
    for mode in ('random', 'scaffold'):
        splits = make_splits(corpus, mode=mode, seed=42)
        try:
            report = assert_split_integrity(corpus, splits, mode)
            sizes = report['sizes']
            detail = (f'train {sizes["train"]} / val {sizes["val"]} / test {sizes["test"]}'
                      + (f', scaffold overlap {report["scaffold_overlap_train_test"]}'
                         if mode == 'scaffold' else ''))
            check(f'{mode} split passes all integrity assertions', True, detail)
            print(f'      dual-labelled per partition: {report["dual_labelled_per_partition"]}')
        except AssertionError as exc:
            check(f'{mode} split passes all integrity assertions', False, str(exc))


def verify_masked_loss(device: torch.device, corpus) -> None:
    section('6. Masked loss and gradient isolation')
    graphs, gstats = build_graph_dataset(corpus, mode='encoded', use_cache=True, verbose=False)

    # Build a batch containing ONLY BBB-labelled molecules (no P-gp labels at all).
    bbb_only = [g for g in graphs if g.m_bbb.item() == 1 and g.m_pgp.item() == 0][:16]
    loader = make_loader(bbb_only, batch_size=16, shuffle=False, device=device)
    batch = next(iter(loader)).to(device)

    tasks = ['bbb', 'pgp']
    model = MultiTaskGNN(
        num_atom_features=gstats['num_atom_features'],
        num_bond_features=gstats['num_bond_features'],
        tasks=tasks, hidden_channels=64, num_layers=2,
    ).to(device)

    cw = {t: torch.tensor(compute_class_weights(np.array([0, 1])), device=device) for t in tasks}
    criterion = MaskedMultiTaskLoss(tasks, cw).to(device)

    targets = {t: getattr(batch, f'y_{t}') for t in tasks}
    masks = {t: getattr(batch, f'm_{t}') for t in tasks}
    check('batch genuinely has no P-gp labels', float(masks['pgp'].sum().item()) == 0.0,
          f'pgp mask sum = {masks["pgp"].sum().item()}')

    logits = model(batch, tasks=tasks)
    loss, components = criterion(logits, targets, masks)
    loss.backward()

    pgp_grads = [p.grad for p in model.heads['pgp'].parameters() if p.grad is not None]
    pgp_grad_norm = float(sum(g.abs().sum().item() for g in pgp_grads)) if pgp_grads else 0.0
    bbb_grads = [p.grad for p in model.heads['bbb'].parameters() if p.grad is not None]
    bbb_grad_norm = float(sum(g.abs().sum().item() for g in bbb_grads)) if bbb_grads else 0.0

    check('unlabelled head receives exactly zero gradient', pgp_grad_norm == 0.0,
          f'P-gp head grad L1 = {pgp_grad_norm:.3e}')
    check('labelled head receives non-zero gradient', bbb_grad_norm > 0.0,
          f'BBB head grad L1 = {bbb_grad_norm:.3e}')
    check('loss component for the unlabelled task is zero', components['pgp'] == 0.0,
          f'components = {{bbb: {components["bbb"]:.4f}, pgp: {components["pgp"]:.4f}}}')

    # Task weight 0 must remove a task entirely, even when it HAS labels.
    dual = [g for g in graphs if g.m_bbb.item() == 1 and g.m_pgp.item() == 1][:16]
    if dual:
        dloader = make_loader(dual, batch_size=16, shuffle=False, device=device)
        dbatch = next(iter(dloader)).to(device)
        model.zero_grad(set_to_none=True)
        crit0 = MaskedMultiTaskLoss(tasks, cw, task_weights={'bbb': 1.0, 'pgp': 0.0}).to(device)
        dt = {t: getattr(dbatch, f'y_{t}') for t in tasks}
        dm = {t: getattr(dbatch, f'm_{t}') for t in tasks}
        check('weight-0 test batch does have P-gp labels', float(dm['pgp'].sum().item()) > 0,
              f'pgp mask sum = {dm["pgp"].sum().item()}')
        l0, c0 = crit0(model(dbatch, tasks=tasks), dt, dm)
        l0.backward()
        g0 = [p.grad for p in model.heads['pgp'].parameters() if p.grad is not None]
        n0 = float(sum(g.abs().sum().item() for g in g0)) if g0 else 0.0
        check('task weight 0 removes the task from the objective', n0 == 0.0,
              f'P-gp head grad L1 = {n0:.3e} despite {int(dm["pgp"].sum().item())} labels present')


def verify_model_shapes(device: torch.device, corpus) -> None:
    section('7. Model forward contract')
    graphs, gstats = build_graph_dataset(corpus, mode='encoded', use_cache=True, verbose=False)
    loader = make_loader(graphs[:8], batch_size=8, shuffle=False, device=device)
    batch = next(iter(loader)).to(device)

    model = MultiTaskGNN(
        num_atom_features=gstats['num_atom_features'],
        num_bond_features=gstats['num_bond_features'],
        tasks=['bbb', 'pgp'], hidden_channels=256, num_layers=3,
    ).to(device)
    out = model(batch)

    check('forward returns one entry per task', set(out.keys()) == {'bbb', 'pgp'}, str(list(out.keys())))
    check('each head emits [batch, 2] logits',
          all(v.shape == (8, 2) for v in out.values()),
          str({k: tuple(v.shape) for k, v in out.items()}))
    check('pooled embedding is [batch, hidden]', model.embed(batch).shape == (8, 256),
          str(tuple(model.embed(batch).shape)))

    meta = model.architecture_metadata()
    check('architecture metadata is self-describing',
          meta['num_atom_features'] == gstats['num_atom_features']
          and meta['trainable_parameters'] == model.count_parameters(),
          f'{meta["trainable_parameters"]:,} parameters')

    single = MultiTaskGNN(
        num_atom_features=gstats['num_atom_features'],
        num_bond_features=gstats['num_bond_features'],
        tasks=['bbb'], hidden_channels=256, num_layers=3,
    )
    delta = model.count_parameters() - single.count_parameters()
    check('second head is cheap', delta < 10000,
          f'+{delta} parameters for the extra head '
          f'({single.count_parameters():,} -> {model.count_parameters():,})')


def main() -> int:
    print('=' * 78)
    print('MULTI-TASK GNN VERIFICATION SUITE')
    print('=' * 78)

    device = verify_device()
    verify_featurizer()
    verify_legacy_parity()
    corpus = verify_corpus()
    verify_splits(corpus)
    verify_masked_loss(device, corpus)
    verify_model_shapes(device, corpus)

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
