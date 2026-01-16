from tqdm import tqdm
import torch
import copy
import pandas as pd
import numpy as np

from utils.metrics import mean_weighted_columnwise_auc


def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_pres_fn,
    loss_loc_fn,
    device,
    alpha_loc: float = 1.0,
):
    """
    Run a single training epoch for the scan-level MIL aggregator.

    Each batch contains a list of scans, where each scan is represented by
    a variable-length set of patch embeddings and patch-level logits.
    The model aggregates patch information into scan-level predictions:
      - a binary aneurysm presence logit
      - multi-label anatomical location logits (13 possible locations)

    The loss is computed conditionally:
      - presence loss is computed for all scans
      - location loss is computed only for scans where an aneurysm is present

    Args:
        model (torch.nn.Module): Scan-level MIL classifier returning:
            z_pres (scalar logit), z_loc (tensor of shape (num_locations,))
        dataloader (DataLoader): Training dataloader yielding batches
            of lists of scan dictionaries.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_pres_fn (callable): Presence loss function
            (e.g. BCEWithLogitsLoss).
        loss_loc_fn (callable): Localization loss function
            (e.g. BCEWithLogitsLoss for multi-label targets).
        device (torch.device): Compute device.
        alpha_loc (float): Weight applied to the localization loss term.
            Defaults to 1.0.

    Returns:
        tuple:
            avg_loss (float): Mean total loss over the epoch.
            avg_loss_pres (float): Mean presence loss.
            avg_loss_loc (float): Mean localization loss
                (computed only on positive scans).
    """
    model.train()
    total_loss = 0.0
    total_loss_pres = 0.0
    total_loss_loc = 0.0

    for batch in tqdm(dataloader, desc="Training"):

        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        batch_loss_pres = 0.0
        batch_loss_loc = 0.0

        for scan in batch:
            embeddings = scan["embeddings"].to(device)
            pres_logits_patch = scan["presence_logits"].to(device)
            loc_logits_patch = scan["location_logits"].to(device)
            centers = scan["centers_norm"].to(device)

            y_pres = scan["y_presence"].to(device)  # scalar float
            y_loc = scan["y_location"].to(device).float()  # list

            # forward pass
            z_pres, z_loc, _, _ = model(
                embeddings, pres_logits_patch, loc_logits_patch, centers
            )

            # Presence loss
            loss_pres = loss_pres_fn(z_pres.unsqueeze(0), y_pres.unsqueeze(0))

            # Location loss (only if aneurysm present)
            if y_pres.item() == 1:
                loss_loc = loss_loc_fn(z_loc, y_loc)
            else:
                loss_loc = torch.tensor(0.0, device=device)

            loss = loss_pres + alpha_loc * loss_loc

            batch_loss += loss
            batch_loss_pres += loss_pres
            batch_loss_loc += loss_loc

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        total_loss_pres += batch_loss_pres.item()
        total_loss_loc += batch_loss_loc.item()

    return (
        total_loss / len(dataloader),
        total_loss_pres / len(dataloader),
        total_loss_loc / len(dataloader),
    )


@torch.no_grad()
def evaluate(
    model, dataloader, loss_pres_fn, loss_loc_fn, device, alpha_loc: float = 1.0
):
    """
    Evaluate the scan-level MIL aggregator on a dataset split.

    Computes:
      - total, presence, and localization losses
      - the challenge metric: Mean Weighted Columnwise AUROC
      - auxiliary metrics: presence AUROC and mean localization AUROC

    Predicted scan-level probabilities are constructed conditionally:
      p(location = l) = p(presence) * p(location = l | presence)

    Args:
        model (torch.nn.Module): Scan-level MIL classifier returning:
            z_pres (scalar logit), z_loc (tensor of shape (num_locations,))
        dataloader (DataLoader): Validation or test dataloader yielding
            batches of lists of scan dictionaries.
        loss_pres_fn (callable): Presence loss function.
        loss_loc_fn (callable): Localization loss function.
        device (torch.device): Compute device.
        alpha_loc (float): Weight applied to the localization loss term.
            Defaults to 1.0.

    Returns:
        tuple:
            avg_loss (float): Mean total loss.
            avg_loss_pres (float): Mean presence loss.
            avg_loss_loc (float): Mean localization loss.
            weighted_auc (float): Challenge metric:
                Mean Weighted Columnwise AUROC.
            presence_auc (float): AUROC for aneurysm presence label.
            localization_auc (float): Mean AUROC across 13 location labels.
    """
    model.eval()

    total_loss = 0.0
    total_loss_pres = 0.0
    total_loss_loc = 0.0

    all_y_true = []
    all_y_pred = []

    for batch in tqdm(dataloader, desc="Validation", leave=False):

        batch_loss = 0.0
        batch_loss_pres = 0.0
        batch_loss_loc = 0.0

        for scan in batch:
            embeddings = scan["embeddings"].to(device)
            pres_logits_patch = scan["presence_logits"].to(device)
            loc_logits_patch = scan["location_logits"].to(device)
            centers = scan["centers_norm"].to(device)

            y_pres = scan["y_presence"].to(device)
            y_loc = scan["y_location"].to(device).float()

            z_pres, z_loc, _, _ = model(
                embeddings, pres_logits_patch, loc_logits_patch, centers
            )

            # Losses
            loss_pres = loss_pres_fn(z_pres.unsqueeze(0), y_pres.unsqueeze(0))

            if y_pres.item() == 1:
                loss_loc = loss_loc_fn(z_loc, y_loc)
            else:
                loss_loc = torch.tensor(0.0, device=device)

            loss = loss_pres + alpha_loc * loss_loc

            batch_loss += loss
            batch_loss_pres += loss_pres
            batch_loss_loc += loss_loc

            # ---- Build conditional probabilities ----
            p_pres = torch.sigmoid(z_pres)  # scalar
            p_loc = torch.sigmoid(z_loc)  # (13,)
            p_loc_joint = p_pres * p_loc  # (13,)

            y_pred_vec = torch.cat([p_loc_joint, p_pres.unsqueeze(0)], dim=0)
            y_true_vec = torch.cat([y_loc, y_pres.unsqueeze(0)], dim=0)

            all_y_true.append(y_true_vec.cpu().numpy())
            all_y_pred.append(y_pred_vec.cpu().numpy())

        total_loss += batch_loss.item()
        total_loss_pres += batch_loss_pres.item()
        total_loss_loc += batch_loss_loc.item()

    avg_loss = total_loss / len(dataloader)
    avg_loss_pres = total_loss_pres / len(dataloader)
    avg_loss_loc = total_loss_loc / len(dataloader)

    # ---- Compute challenge metric ----
    y_true_np = np.stack(all_y_true)
    y_pred_np = np.stack(all_y_pred)

    weighted_auc, per_label_auc = mean_weighted_columnwise_auc(y_true_np, y_pred_np)

    presence_auc = per_label_auc[13]
    localization_auc = np.nanmean(per_label_auc[:13])

    return (
        avg_loss,
        avg_loss_pres,
        avg_loss_loc,
        weighted_auc,
        presence_auc,
        localization_auc,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    loss_pres_fn,
    loss_loc_fn,
    optimizer,
    epochs,
    patience,
    device,
    output_dir,
    alpha_loc=1.0,
):
    """
    Train the scan-level MIL aggregator with early stopping and checkpointing.

    The model learns to aggregate variable-length patch representations
    into scan-level predictions for:
      - aneurysm presence (binary)
      - aneurysm anatomical locations (multi-label)

    Optimization objective:
      - presence loss on all scans
      - localization loss on positive scans only
      - early stopping based on the challenge metric
        (Mean Weighted Columnwise AUROC)

    Logged metrics per epoch:
      - total, presence, and localization losses
      - weighted challenge AUROC
      - presence AUROC
      - mean localization AUROC

    The best model is selected based on highest validation weighted AUROC.

    Args:
        model (torch.nn.Module): Scan-level MIL classifier.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        loss_pres_fn (callable): Presence loss function.
        loss_loc_fn (callable): Localization loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        device (torch.device): Compute device.
        output_dir (Path): Directory for saving checkpoints and metrics.
        alpha_loc (float): Weight applied to localization loss. Defaults to 1.0.

    Returns:
        tuple:
            best_model_wts (dict): State dict of the best-performing model.
            metrics (list[dict]): Per-epoch training and validation metrics.
    """
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = -float("inf")
    epochs_since_best = 0
    metrics = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_loss_pres, train_loss_loc = train_epoch(
            model, train_loader, optimizer, loss_pres_fn, loss_loc_fn, device, alpha_loc
        )

        (
            val_loss,
            val_loss_pres,
            val_loss_loc,
            val_weighted_auc,
            val_presence_auc,
            val_localization_auc,
        ) = evaluate(model, val_loader, loss_pres_fn, loss_loc_fn, device, alpha_loc)

        metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_loss_pres": train_loss_pres,
                "train_loss_loc": train_loss_loc,
                "val_loss": val_loss,
                "val_loss_pres": val_loss_pres,
                "val_loss_loc": val_loss_loc,
                "val_weighted_auc": val_weighted_auc,
                "val_presence_auc": val_presence_auc,
                "val_localization_auc": val_localization_auc,
            }
        )

        print(
            f"Train | Loss: {train_loss:.4f} "
            f"(Pres: {train_loss_pres:.4f}, Loc: {train_loss_loc:.4f})"
        )
        print(
            f"Val   | Loss: {val_loss:.4f} "
            f"(Pres: {val_loss_pres:.4f}, Loc: {val_loss_loc:.4f})"
        )
        print(
            f"Val Weighted AUC: {val_weighted_auc:.4f} | "
            f"Presence AUC: {val_presence_auc:.4f} | "
            f"Localization Mean AUC: {val_localization_auc:.4f}"
        )

        pd.DataFrame(metrics).to_csv(output_dir / "training_metrics.csv", index=False)

        # Save best model based on challenge metric
        if val_weighted_auc > best_metric:
            best_metric = val_weighted_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), output_dir / "weights.pth")
            print("Model saved (best weighted AUC).")
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epochs_since_best >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

    return best_model_wts, metrics
