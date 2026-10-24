import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, average="macro"),
        "recall": recall_score(y_true, y_pred, zero_division=0, average="macro"),
        "f1": f1_score(y_true, y_pred, zero_division=0, average="macro"),
    }


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
):
    model.train()

    running_loss = 0.0
    count = 0

    all_targets = []
    all_preds = []

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        speech = batch["speech"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(speech)

        loss = criterion(logits, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        batch_size = speech.shape[0]
        running_loss += loss.item() * batch_size
        count += batch_size

        all_targets.append(batch["target"].numpy())
        all_preds.append(logits.detach().cpu().numpy().argmax(axis=1))

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(count, 1)

    metrics = classification_metrics(
        np.concatenate(all_targets),
        np.concatenate(all_preds),
    )

    return avg_loss, metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):
    model.eval()

    running_loss = 0.0
    count = 0

    all_targets = []
    all_preds = []

    pbar = tqdm(loader, desc="Validation", leave=False)

    for batch in pbar:
        speech = batch["speech"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        logits = model(speech)

        loss = criterion(logits, target)

        batch_size = speech.shape[0]
        running_loss += loss.item() * batch_size
        count += batch_size

        all_targets.append(batch["target"].numpy())
        all_preds.append(logits.cpu().numpy().argmax(axis=1))

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(count, 1)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    metrics = classification_metrics(
        y_true,
        y_pred,
    )

    return (avg_loss, metrics, y_true, y_pred)


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epochs: int,
    checkpoint: str | None = None,
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 70}")

        print(f"Epoch " f"{epoch}/{epochs}")

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        (val_loss, val_metrics, val_true, val_prob) = validate(
            model=model, loader=val_loader, criterion=criterion, device=device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])

        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])

        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"train loss={train_loss:.4f} "
            f"| train F1={train_metrics['f1']:.4f} "
            f"| train P={train_metrics['precision']:.4f} "
            f"| train R={train_metrics['recall']:.4f}"
        )

        print(
            f"val   loss={val_loss:.4f} "
            f"| val   F1={val_metrics['f1']:.4f} "
            f"| val   P={val_metrics['precision']:.4f} "
            f"| val   R={val_metrics['recall']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]

            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

            if checkpoint is not None:
                torch.save({"model_state_dict": best_state}, checkpoint)

                print(f"Saved best model. " f"val F1={best_val_f1:.4f}")

    model.load_state_dict(best_state)

    return history
