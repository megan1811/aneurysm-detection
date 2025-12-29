from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import torch
import copy
import pandas as pd


def train_epoch(model, dataloader, optimizer, loss_pres_fn, loss_loc_fn, device):
    """
    Run a single training epoch for the two-head patch classifier.

    Performs a single pass over the training dataloader, updates model weights,
    and computes patch-level metrics. The presence head is trained on all samples,
    while the location head is trained only on positive patches via masking.

    Args:
        model (torch.nn.Module): Patch classifier returning
            (pres_logits: (B,), loc_logits: (B, num_loc_classes)).
        dataloader (DataLoader): Training dataloader yielding patch batches.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_pres_fn (callable): Presence loss (e.g. BCEWithLogitsLoss).
        loss_loc_fn (callable): Location loss (e.g. CrossEntropyLoss).
        device (torch.device): Compute device.

    Returns:
        tuple:
            avg_loss (float): Mean total loss over the epoch.
            avg_loss_pres (float): Mean presence loss.
            avg_loss_loc (float): Mean location loss (positives only).
            auc_pres (float): Presence AUROC.
            loc_acc (float): Location accuracy on positives.
            loc_bal_acc (float): Balanced location accuracy on positives.
    """
    model.train()

    running_loss = 0.0
    running_loss_pres = 0.0
    running_loss_loc = 0.0

    all_preds_pres = []
    all_labels_pres = []

    all_preds_loc = []
    all_labels_loc = []

    for batch in tqdm(dataloader, desc="Training"):
        patches = batch["patch"].to(device)
        coords = batch["coords"].to(device)
        modality = batch["modality"].to(device)

        y_pres = batch["y_pres"].to(device)  # float32 (B,)
        y_loc = batch["y_loc"].to(device)  # long (B,)

        optimizer.zero_grad(set_to_none=True)

        pres_logits, loc_logits = model(patches, coords, modality)

        loss_pres = loss_pres_fn(pres_logits.view(-1), y_pres.view(-1))

        pos_mask = y_pres.view(-1) == 1
        if pos_mask.any():
            loss_loc = loss_loc_fn(loc_logits[pos_mask], y_loc[pos_mask])
        else:
            loss_loc = torch.tensor(0.0, device=device)

        loss = loss_pres + loss_loc

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_pres += loss_pres.item()
        running_loss_loc += loss_loc.item()

        # Metrics accumulation
        p_pres = torch.sigmoid(pres_logits).detach().cpu().numpy()
        all_preds_pres.extend(p_pres.tolist())
        all_labels_pres.extend(y_pres.detach().cpu().numpy().astype(int).tolist())

        if pos_mask.any():
            preds_loc = loc_logits[pos_mask].argmax(dim=1).detach().cpu().numpy()
            all_preds_loc.extend(preds_loc.tolist())
            all_labels_loc.extend(y_loc[pos_mask].detach().cpu().numpy().tolist())

    avg_loss = running_loss / len(dataloader)
    avg_loss_pres = running_loss_pres / len(dataloader)
    avg_loss_loc = running_loss_loc / len(dataloader)

    # Presence AUROC
    try:
        auc_pres = roc_auc_score(all_labels_pres, all_preds_pres)
    except ValueError:
        auc_pres = float("nan")

    # Location metrics (positives only)
    if len(all_labels_loc) > 0:
        loc_acc = accuracy_score(all_labels_loc, all_preds_loc)
        loc_bal_acc = balanced_accuracy_score(all_labels_loc, all_preds_loc)
    else:
        loc_acc = float("nan")
        loc_bal_acc = float("nan")

    return avg_loss, avg_loss_pres, avg_loss_loc, auc_pres, loc_acc, loc_bal_acc


def eval(model, dataloader, loss_pres_fn, loss_loc_fn, device):
    """
    Evaluate the two-head patch classifier on a dataset split.

    Runs a forward-only pass over the dataloader and computes the same losses
    and metrics as in training. Location loss and metrics are computed only
    on positive patches via masking.

    Args:
        model (torch.nn.Module): Patch classifier returning
            (pres_logits, loc_logits).
        dataloader (DataLoader): Validation or test dataloader.
        loss_pres_fn (callable): Presence loss.
        loss_loc_fn (callable): Location loss.
        device (torch.device): Compute device.

    Returns:
        tuple:
            avg_loss (float): Mean total evaluation loss.
            avg_loss_pres (float): Mean presence loss.
            avg_loss_loc (float): Mean location loss (positives only).
            auc_pres (float): Presence AUROC.
            loc_acc (float): Location accuracy on positives.
            loc_bal_acc (float): Balanced location accuracy on positives.
    """
    model.eval()
    running_loss = 0.0
    running_loss_pres = 0.0
    running_loss_loc = 0.0

    all_preds_pres = []
    all_labels_pres = []

    all_preds_loc = []
    all_labels_loc = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            patches = batch["patch"].to(device)
            coords = batch["coords"].to(device)
            modality = batch["modality"].to(device)

            y_pres = batch["y_pres"].to(device)
            y_loc = batch["y_loc"].to(device)

            pres_logits, loc_logits = model(patches, coords, modality)

            loss_pres = loss_pres_fn(pres_logits.view(-1), y_pres.view(-1))

            pos_mask = y_pres.view(-1) == 1
            if pos_mask.any():
                loss_loc = loss_loc_fn(loc_logits[pos_mask], y_loc[pos_mask])
            else:
                loss_loc = torch.tensor(0.0, device=device)

            loss = loss_pres + loss_loc

            running_loss += loss.item()
            running_loss_pres += loss_pres.item()
            running_loss_loc += loss_loc.item()

            p_pres = torch.sigmoid(pres_logits).cpu().numpy()
            all_preds_pres.extend(p_pres.tolist())
            all_labels_pres.extend(y_pres.cpu().numpy().astype(int).tolist())

            if pos_mask.any():
                preds_loc = loc_logits[pos_mask].argmax(dim=1).cpu().numpy()
                all_preds_loc.extend(preds_loc.tolist())
                all_labels_loc.extend(y_loc[pos_mask].cpu().numpy().tolist())

    avg_loss = running_loss / len(dataloader)
    avg_loss_pres = running_loss_pres / len(dataloader)
    avg_loss_loc = running_loss_loc / len(dataloader)

    try:
        auc_pres = roc_auc_score(all_labels_pres, all_preds_pres)
    except ValueError:
        auc_pres = float("nan")

    if len(all_labels_loc) > 0:
        loc_acc = accuracy_score(all_labels_loc, all_preds_loc)
        loc_bal_acc = balanced_accuracy_score(all_labels_loc, all_preds_loc)
    else:
        loc_acc = float("nan")
        loc_bal_acc = float("nan")

    return avg_loss, avg_loss_pres, avg_loss_loc, auc_pres, loc_acc, loc_bal_acc


def train_model(
    model,
    device,
    train_loader,
    val_loader,
    loss_pres_fn,
    loss_loc_fn,
    optimizer,
    scheduler,
    epochs,
    patience,
    output_dir,
):
    """
    Train a two-head patch classifier with early stopping and checkpointing.

    The model jointly optimizes:
      - a binary presence head (aneurysm present vs not)
      - a multiclass location head (anatomical location), trained only on
        positive samples via masking.

    Training and validation metrics are computed at the patch level:
      - Presence: AUROC
      - Location: accuracy and balanced accuracy on positive patches only

    The best model is selected based on minimum validation total loss
    (presence loss + location loss).

    Args:
        model (torch.nn.Module): Patch classifier returning
            (pres_logits, loc_logits).
        device (torch.device): Compute device.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        loss_pres_fn (callable): Presence loss.
        loss_loc_fn (callable): Location loss.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): LR scheduler stepped on val loss.
        epochs (int): Max number of epochs.
        patience (int): Early stopping patience.
        output_dir (Path): Directory for checkpoints and metrics.

    Returns:
        tuple:
            - best_model_wts (dict): State dict of the best-performing model.
            - metrics (list[dict]): Per-epoch training and validation metrics.
    """
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_since_best = 0
    metrics = []

    for epoch in range(epochs):
        (
            train_loss,
            train_loss_pres,
            train_loss_loc,
            train_auc_pres,
            train_loc_acc,
            train_loc_bal_acc,
        ) = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_pres_fn,
            loss_loc_fn,
            device,
        )

        (
            val_loss,
            val_loss_pres,
            val_loss_loc,
            val_auc_pres,
            val_loc_acc,
            val_loc_bal_acc,
        ) = eval(
            model,
            val_loader,
            loss_pres_fn,
            loss_loc_fn,
            device,
        )
        metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_loss_pres": train_loss_pres,
                "train_loss_loc": train_loss_loc,
                "train_auc_pres": train_auc_pres,
                "train_loc_acc": train_loc_acc,
                "train_loc_bal_acc": train_loc_bal_acc,
                "val_loss": val_loss,
                "val_loss_pres": val_loss_pres,
                "val_loss_loc": val_loss_loc,
                "val_auc_pres": val_auc_pres,
                "val_loc_acc": val_loc_acc,
                "val_loc_bal_acc": val_loc_bal_acc,
            }
        )

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(
            f"Train | Loss: {train_loss:.4f} "
            f"(Pres: {train_loss_pres:.4f}, Loc: {train_loss_loc:.4f}) | "
            f"AUC: {train_auc_pres:.4f} | "
            f"Loc Acc: {train_loc_acc:.4f}"
        )
        print(
            f"Val   | Loss: {val_loss:.4f} "
            f"(Pres: {val_loss_pres:.4f}, Loc: {val_loss_loc:.4f}) | "
            f"AUC: {val_auc_pres:.4f} | "
            f"Loc Acc: {val_loc_acc:.4f}"
        )

        # save trainign metrics to csv
        pd.DataFrame(metrics).to_csv(output_dir / "training_metrics.csv", index=False)

        # Checkpoint and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), output_dir / "weights.pth")
            print("model saved.")
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Early stopping
        if epochs_since_best >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

    return best_model_wts, metrics
