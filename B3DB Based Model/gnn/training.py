"""
Shared Training Loop for the Multi-Task GNN.

Extracted from scripts/train_multitask_gnn.py so the single-split runner, the
K-fold runner and the seed-variance runner all use one implementation. Three
copies of an early-stopping loop would drift apart exactly the way the base
paper's two featurizers did.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .model import MultiTaskGNN
from .losses import MaskedMultiTaskLoss, build_class_weights
from .dataset import make_loader

ALL_TASKS = ('bbb', 'pgp')


@dataclass
class TrainConfig:
    """Every knob the training loop reads. Defaults match the reference implementation."""
    tasks: List[str] = field(default_factory=lambda: ['bbb', 'pgp'])
    task_weights: Dict[str, float] = field(default_factory=dict)
    epochs: int = 200
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 5e-4
    hidden: int = 256
    dropout: float = 0.30
    layers: int = 3
    patience: int = 20
    clip_grad: float = 2.0
    seed: int = 42
    residual: bool = False
    amp: bool = False

    def resolved_weights(self) -> Dict[str, float]:
        w = {t: 1.0 for t in self.tasks}
        w.update({k: v for k, v in (self.task_weights or {}).items() if k in w})
        return w

    def active_tasks(self) -> List[str]:
        w = self.resolved_weights()
        return [t for t in self.tasks if w[t] > 0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def set_all_seeds(seed: int) -> None:
    """Seeds python, numpy and torch, including all CUDA devices."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_task_weights(spec: str, tasks: List[str]) -> Dict[str, float]:
    """Parses a 'pgp=0,bbb=2' style override string."""
    weights = {t: 1.0 for t in tasks}
    if not spec:
        return weights
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f'Malformed task-weight entry {part!r}; expected name=value')
        name, value = part.split('=', 1)
        name = name.strip()
        if name not in tasks:
            raise ValueError(f'Task weight names {name!r}, not an active task {tasks}')
        weights[name] = float(value)
    return weights


def move_batch(batch, device: torch.device):
    return batch.to(device, non_blocking=(device.type == 'cuda'))


def run_epoch(
    model: MultiTaskGNN,
    loader,
    criterion: MaskedMultiTaskLoss,
    device: torch.device,
    tasks: List[str],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    clip_grad: float = 0.0,
) -> Dict[str, Any]:
    """One pass over a loader. Trains when an optimizer is supplied, else evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    n_batches = 0
    collected: Dict[str, Dict[str, List]] = {
        t: {'y': [], 'p': [], 'prob': [], 'idx': []} for t in tasks
    }

    for batch in loader:
        batch = move_batch(batch, device)
        targets = {t: getattr(batch, f'y_{t}') for t in tasks}
        masks = {t: getattr(batch, f'm_{t}') for t in tasks}

        with torch.set_grad_enabled(training):
            if scaler is not None and training:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(batch, tasks=tasks)
                    loss, _ = criterion(logits, targets, masks)
            else:
                logits = model(batch, tasks=tasks)
                loss, _ = criterion(logits, targets, masks)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()

        total_loss += float(loss.detach().item())
        n_batches += 1

        # Collect predictions only for molecules that actually carry each label.
        for t in tasks:
            sel = masks[t].bool()
            if sel.sum() == 0:
                continue
            lg = logits[t][sel].detach().float()
            prob = torch.softmax(lg, dim=1)[:, 1]
            collected[t]['y'].extend(targets[t][sel].cpu().numpy().tolist())
            collected[t]['p'].extend(lg.argmax(dim=1).cpu().numpy().tolist())
            collected[t]['prob'].extend(prob.cpu().numpy().tolist())
            collected[t]['idx'].extend(batch.corpus_row[sel].cpu().numpy().tolist())

    out: Dict[str, Any] = {'loss': total_loss / max(n_batches, 1), 'per_task': {}}
    for t in tasks:
        c = collected[t]
        if c['y']:
            out['per_task'][t] = {
                'y_true': np.array(c['y']),
                'y_pred': np.array(c['p']),
                'y_prob': np.array(c['prob']),
                'corpus_row': np.array(c['idx']),
            }
    return out


def quick_f1(pack: Dict[str, Any]) -> float:
    return float(f1_score(pack['y_true'], pack['y_pred'], zero_division=0))


def has_active_label(graph, active_tasks: List[str]) -> bool:
    return any(float(getattr(graph, f'm_{t}').item()) > 0 for t in active_tasks)


def train_one_model(
    train_graphs: List[Any],
    val_graphs: List[Any],
    corpus: pd.DataFrame,
    train_corpus_rows,
    cfg: TrainConfig,
    device: torch.device,
    num_atom_features: int,
    num_bond_features: int,
    verbose: bool = False,
) -> Tuple[MultiTaskGNN, Dict[str, Any]]:
    """
    Trains one model to convergence with early stopping, returned with its
    best-validation weights loaded.

    Molecules carrying no label for any active task are dropped from TRAIN only:
    they contribute exactly zero loss, so this is a speed optimisation and never a
    change of evaluation. Validation is left complete.

    Early stopping, LR scheduling and checkpointing are all driven by the MEAN
    validation F1 across active tasks. With two tasks that scalar has to be stated
    explicitly; the reference implementation leaves it implicit with one task.
    """
    set_all_seeds(cfg.seed)
    tasks = list(cfg.tasks)
    weights = cfg.resolved_weights()
    active = cfg.active_tasks()
    if not active:
        raise ValueError('All task weights are zero; nothing to train.')

    train_graphs = [g for g in train_graphs if has_active_label(g, active)]

    train_loader = make_loader(train_graphs, cfg.batch_size, shuffle=True, device=device)
    val_loader = make_loader(val_graphs, cfg.batch_size, shuffle=False, device=device)

    model = MultiTaskGNN(
        num_atom_features=num_atom_features,
        num_bond_features=num_bond_features,
        tasks=tasks,
        hidden_channels=cfg.hidden,
        dropout=cfg.dropout,
        num_layers=cfg.layers,
        residual=cfg.residual,
    ).to(device)

    class_weights = build_class_weights(corpus, tasks, train_corpus_rows, device)
    criterion = MaskedMultiTaskLoss(tasks, class_weights, weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler(device.type) if (cfg.amp and device.type == 'cuda') else None

    best_score = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    stale = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, tasks,
                       optimizer=optimizer, scaler=scaler, clip_grad=cfg.clip_grad)
        va = run_epoch(model, val_loader, criterion, device, tasks)

        task_f1 = {t: quick_f1(va['per_task'][t]) for t in active if t in va['per_task']}
        score = float(np.mean(list(task_f1.values()))) if task_f1 else 0.0
        scheduler.step(score)

        history.append({
            'epoch': epoch,
            'train_loss': round(tr['loss'], 5),
            'val_loss': round(va['loss'], 5),
            'val_score': round(score, 5),
            **{f'val_f1_{t}': round(v, 5) for t, v in task_f1.items()},
            'lr': optimizer.param_groups[0]['lr'],
        })

        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            detail = ' '.join(f'{t}_f1={v:.4f}' for t, v in task_f1.items())
            print(f'    epoch {epoch:>3}  train_loss={tr["loss"]:.4f}  '
                  f'val_loss={va["loss"]:.4f}  {detail}  score={score:.4f}')

        if stale >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    info: Dict[str, Any] = {
        'best_epoch': best_epoch,
        'best_val_score': round(float(best_score), 5),
        'epochs_run': len(history),
        'history': history,
        'train_molecules': len(train_graphs),
        'class_weights': {t: class_weights[t].cpu().numpy().round(4).tolist()
                          for t in class_weights},
        'criterion': criterion,
    }
    return model, info


def predict_graphs(
    model: MultiTaskGNN,
    graphs: List[Any],
    criterion: MaskedMultiTaskLoss,
    device: torch.device,
    tasks: List[str],
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Runs inference over a graph list and returns per-task prediction packs."""
    loader = make_loader(graphs, batch_size, shuffle=False, device=device)
    return run_epoch(model, loader, criterion, device, tasks)
