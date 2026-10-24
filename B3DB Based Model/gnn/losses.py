"""
Masked Multi-Task Loss.

Only ~216 of ~8,800 molecules in the merged corpus carry both labels, so roughly
97.5% have exactly one. A plain summed cross-entropy over two heads would train
each head on fabricated labels for the molecules it has no ground truth for.

Each head's loss is therefore computed over only its labelled molecules and
normalised by that head's own mask count, so the 7,805-molecule BBB task cannot
swamp the 1,212-molecule P-gp task purely by weight of numbers. Per-task inverse
frequency class weights are preserved from the reference implementation.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F


def compute_class_weights(labels: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Inverse-frequency class weights normalised to mean 1.

    Same formula as the reference implementation: total / (num_classes * count).
    Absent classes get weight 1.0 rather than inf.
    """
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    total = float(len(labels))
    weights = np.ones(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    return weights.astype(np.float32)


class MaskedMultiTaskLoss(torch.nn.Module):
    """
    Sums per-task masked cross-entropy with configurable task weights.

    Setting a task's weight to 0.0 removes it from the objective entirely, which
    is how the multi-task plumbing is verified against the single-task baseline.
    """

    def __init__(
        self,
        tasks: List[str],
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.task_weights = {t: 1.0 for t in self.tasks}
        if task_weights:
            self.task_weights.update(task_weights)

        for task in self.tasks:
            w = (class_weights or {}).get(task)
            if w is not None:
                # Registered as a buffer so .to(device) carries it to the GPU with
                # the rest of the module.
                self.register_buffer(f'class_weight_{task}', w)

    def _weight_for(self, task: str) -> Optional[torch.Tensor]:
        return getattr(self, f'class_weight_{task}', None)

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        logits  : {task: [B, C]}
        targets : {task: [B] int64}
        masks   : {task: [B] with 1 where the label is real}

        Returns (total_loss, per_task_loss_values). The total is a tensor that
        carries gradient; the per-task values are plain floats for logging.
        """
        total = None
        components: Dict[str, float] = {}

        for task in self.tasks:
            weight = self.task_weights.get(task, 1.0)
            if task not in logits or weight == 0.0:
                components[task] = 0.0
                continue

            mask = masks[task].bool()
            n_labelled = int(mask.sum().item())
            if n_labelled == 0:
                components[task] = 0.0
                continue

            task_logits = logits[task][mask]
            task_targets = targets[task][mask].long()

            # reduction='mean' divides by this head's own labelled count, which is
            # the normalisation that stops the larger task dominating.
            loss = F.cross_entropy(
                task_logits, task_targets, weight=self._weight_for(task), reduction='mean'
            )
            components[task] = float(loss.detach().item())
            scaled = weight * loss
            total = scaled if total is None else total + scaled

        if total is None:
            # No labelled molecules for any active task in this batch. Return a
            # real zero that still participates in autograd.
            any_logit = next(iter(logits.values()))
            total = (any_logit.sum() * 0.0)

        return total, components


def build_class_weights(
    corpus,
    tasks: List[str],
    train_idx,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Per-task class weights computed from the TRAIN partition only.

    Computing these over the whole corpus would leak test-set class balance into
    training, which is the kind of subtle leak that survives review unnoticed.
    """
    weights: Dict[str, torch.Tensor] = {}
    train = corpus.iloc[train_idx]
    for task in tasks:
        labelled = train[train[f'm_{task}'] == 1]
        if len(labelled) == 0:
            continue
        w = compute_class_weights(labelled[f'y_{task}'].values)
        weights[task] = torch.tensor(w, dtype=torch.float32, device=device)
    return weights
