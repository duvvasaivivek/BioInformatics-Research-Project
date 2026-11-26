"""Independent GINE models, resumable fitting, and held-out calibration."""

import io
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.special import expit
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from .dataset import ATOM_DIM, BOND_DIM, graph_data, load_dataset, load_torch, save_torch
from .sources import digest
from .storage import ROOT, write_json

MODELS = ROOT / "models"
MODEL_VERSION = 1
SEEDS = (11, 23, 37, 53, 71)


class GINE(nn.Module):
    def __init__(self):
        super().__init__()
        width = 128
        self.project = nn.Linear(ATOM_DIM, width)
        self.layers = nn.ModuleList(
            GINEConv(nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)),
                     train_eps=True, edge_dim=BOND_DIM) for _ in range(3)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.layers)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, batch):
        x = self.project(batch.x).relu()
        for layer, norm in zip(self.layers, self.norms):
            x = norm(layer(x, batch.edge_index, batch.edge_attr)).relu()
        return self.head(global_mean_pool(x, batch.batch)).flatten()


def device_for(choice):
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this interpreter. Use --device cpu or install a CUDA-enabled PyTorch build.")
    device = torch.device("cuda" if torch.cuda.is_available() and choice != "cpu" else "cpu")
    # Small molecular graphs do poorly under a large CPU thread pool.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return device


def checkpoint_path(endpoint, seed):
    return MODELS / endpoint / f"seed_{seed}.pt"


def data_loader(cache, endpoint, keys, batch_size, seed=None):
    dataset = [graph_data(cache, key, endpoint) for key in keys]
    return DataLoader(dataset, batch_size=batch_size, shuffle=seed is not None, num_workers=0,
                      generator=torch.Generator().manual_seed(0 if seed is None else seed))


@torch.inference_mode()
def predict_logits(model, loader, device):
    model.eval()
    parts = [model(batch.to(device)).cpu().numpy() for batch in loader]
    logits = np.concatenate(parts) if parts else np.empty(0)
    if not np.isfinite(logits).all():
        raise ValueError("The model produced non-finite logits.")
    return logits


def train_seed(cache, endpoint, seed, epochs, patience, batch_size, device):
    path = checkpoint_path(endpoint, seed)
    config = {"model_version": MODEL_VERSION, "data_id": cache["data_id"], "endpoint": endpoint,
              "seed": seed, "epochs": epochs, "patience": patience, "batch_size": batch_size}
    checkpoint = load_torch(path) if path.exists() else None
    if checkpoint is not None and checkpoint["config"] != config:
        raise ValueError(f"{path} belongs to a different dataset or training configuration. "
                         "Restore the original settings or move this checkpoint aside.")
    if checkpoint is not None and checkpoint["complete"]:
        print(f"[{endpoint} seed {seed}] Already complete at epoch {checkpoint['epoch']}.", flush=True)
        return

    torch.manual_seed(seed)
    model = GINE().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    parts = cache["splits"][endpoint]
    labels = cache["labels"][endpoint]
    validation = data_loader(cache, endpoint, parts["validation"], batch_size)
    validation_labels = [labels[key] for key in parts["validation"]]
    positive = sum(labels[key] for key in parts["train"])
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(parts["train"]) - positive) / positive, device=device))
    history, best_epoch, best_ap, best_weights = [], 0, -1.0, None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["weights"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        history, best_epoch, best_ap, best_weights = (
            checkpoint[name] for name in ("history", "best_epoch", "best_ap", "best_weights"))
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device=device)

    print(f"[{endpoint} seed {seed}] {device.type.upper()} training from epoch {len(history) + 1}.", flush=True)
    for epoch in range(len(history) + 1, epochs + 1):
        training = data_loader(cache, endpoint, parts["train"], batch_size, seed + epoch)
        model.train()
        loss_sum = torch.zeros((), device=device)
        for batch in training:
            batch = batch.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            optimiser.step()
            loss_sum += loss.detach() * batch.num_graphs
        mean_loss = float(loss_sum.cpu()) / len(parts["train"])
        logits = predict_logits(model, validation, device)
        if not np.isfinite(mean_loss):
            raise ValueError("Training produced a non-finite loss. The previous checkpoint was retained.")
        ap = float(average_precision_score(validation_labels, logits))
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": mean_loss, "validation_ap": ap})
        complete = epoch == epochs or epoch - best_epoch >= patience
        save_torch(path, {
            "config": config, "complete": complete, "epoch": epoch, "weights": model.state_dict(),
            "optimiser": optimiser.state_dict(), "history": history, "best_epoch": best_epoch,
            "best_ap": best_ap, "best_weights": best_weights, "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        })
        print(f"[{endpoint} seed {seed}] epoch {epoch:03d}  loss {mean_loss:.4f}  val AP {ap:.4f} "
              f"(best {best_ap:.4f} at {best_epoch})", flush=True)
        if complete:
            break


def train(endpoint, seeds, epochs, patience, batch_size, choice):
    cache = load_dataset()
    device = device_for(choice)
    for seed in seeds:
        train_seed(cache, endpoint, seed, epochs, patience, batch_size, device)


def load_ensemble(cache, endpoint, seeds, device):
    models, receipts = [], {}
    for seed in seeds:
        path = checkpoint_path(endpoint, seed)
        if not path.exists():
            raise ValueError(f"Missing {endpoint} seed {seed}. Run 'train {endpoint}' with the matching seeds.")
        raw = path.read_bytes()
        checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if (config["data_id"], config["model_version"], config["endpoint"], config["seed"]) != (
                cache["data_id"], MODEL_VERSION, endpoint, seed):
            raise ValueError(f"Checkpoint {path} has incompatible data, model version, or identity.")
        if not checkpoint["complete"]:
            raise ValueError(f"Resume {endpoint} seed {seed} to finish training.")
        model = GINE().to(device)
        model.load_state_dict(checkpoint["best_weights"])
        models.append(model)
        receipts[str(seed)] = digest(raw)
    return models, receipts


def ensemble_logits(cache, endpoint, keys, models, device):
    loader = data_loader(cache, endpoint, keys, 128)
    return np.stack([predict_logits(model, loader, device) for model in models]).astype(np.float64)


def calibrated_probability(logits, calibration):
    return expit(calibration["slope"] * logits + calibration["intercept"])


def probabilities(logits, calibration):
    centre = calibrated_probability(logits.mean(axis=0), calibration)
    members = calibrated_probability(logits, calibration)
    return centre, members.std(axis=0)


def sensitivity_threshold(y, p, target):
    positives = np.sort(np.asarray(p)[np.asarray(y) == 1])[::-1]
    if not len(positives):
        raise ValueError("Threshold selection requires positives.")
    return float(positives[int(np.ceil(target * len(positives))) - 1])


def metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p)
    pred = p >= threshold
    tp, fp = int(sum(pred & (y == 1))), int(sum(pred & (y == 0)))
    tn, fn = int(sum(~pred & (y == 0))), int(sum(~pred & (y == 1)))
    both = len(set(y)) == 2
    return {"n": len(y), "positive": int(y.sum()), "ap": float(average_precision_score(y, p)) if both else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None}


def fit_score_calibration(logits, labels, min_sensitivity):
    classifier = LogisticRegression().fit(logits.reshape(-1, 1), labels)
    report = {"slope": float(classifier.coef_[0, 0]), "intercept": float(classifier.intercept_[0]),
              "target_sensitivity": min_sensitivity}
    probs = expit(report["slope"] * logits + report["intercept"])
    report["threshold"] = sensitivity_threshold(labels, probs, min_sensitivity)
    report["calibration_metrics"] = metrics(labels, probs, report["threshold"])
    return report


def calibrate(endpoint, seeds, min_sensitivity, choice):
    cache = load_dataset()
    device = device_for(choice)
    models, receipts = load_ensemble(cache, endpoint, seeds, device)
    keys = cache["splits"][endpoint]["calibration"]
    labels = [cache["labels"][endpoint][key] for key in keys]
    logits = ensemble_logits(cache, endpoint, keys, models, device)
    report = {"data_id": cache["data_id"], "endpoint": endpoint, "model_version": MODEL_VERSION,
              "checkpoint_sha256": receipts, "seeds": list(seeds),
              **fit_score_calibration(logits.mean(axis=0), labels, min_sensitivity)}
    path = MODELS / endpoint / "calibration.json"
    write_json(path, report)
    print(f"[{endpoint}] threshold={report['threshold']:.6g}  calibration metrics={report['calibration_metrics']}")
    print(path)


def calibrated_ensemble(cache, endpoint, device):
    path = MODELS / endpoint / "calibration.json"
    if not path.exists():
        raise ValueError(f"Run 'calibrate {endpoint}' first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["data_id"] != cache["data_id"]:
        raise ValueError(f"Stale {endpoint} calibration. Restore its matching data snapshot or train and calibrate "
                         "new checkpoints on the current dataset.")
    if (report["endpoint"], report["model_version"]) != (endpoint, MODEL_VERSION):
        raise ValueError(f"Stale {endpoint} calibration. Refit it using the current checkpoints.")
    models, receipts = load_ensemble(cache, endpoint, report["seeds"], device)
    if receipts != report["checkpoint_sha256"]:
        raise ValueError(f"{endpoint} checkpoints changed. Rerun calibration with the matching seeds.")
    return models, report


def score(cache, endpoint, keys, ensemble, device):
    models, calibration = ensemble
    logits = ensemble_logits(cache, endpoint, keys, models, device)
    p, sd = probabilities(logits, calibration)
    return {key: (float(prob), float(spread)) for key, prob, spread in zip(keys, p, sd)}
