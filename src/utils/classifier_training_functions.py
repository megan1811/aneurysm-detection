from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import copy
import pandas as pd


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Run a single training epoch for the patch classifier.

    Iterates once over the training dataloader, performs forward and backward
    passes, updates the model parameters, and computes both standard and
    balanced accuracy over the epoch.

    Args:
        model (torch.nn.Module): Patch-level classification model. Expected
            forward signature: ``model(patches, coords, modality)`` returning
            logits of shape (B, num_classes).
        dataloader (torch.utils.data.DataLoader): Dataloader yielding batches
            of dictionaries with keys:
                - "patch": image or volume patches of shape (B, C, H, W, ...)
                - "coords": coordinate tensor of shape (B, D_coord)
                - "modality": modality encoding tensor of shape (B, D_mod)
                - "y": binary label aneurysm present of shape (1,)
                - "location" (torch.Tensor, optional): Encoded anatomical location of the aneurysm of shape (1,)
        optimizer (torch.optim.Optimizer): Optimizer used to update model
            parameters.
        criterion (callable): Loss function taking (logits, targets) and
            returning a scalar loss, e.g. ``nn.CrossEntropyLoss``.
        device (torch.device): Device on which computations are performed.

    Returns:
        tuple[float, float, float]:
            - avg_loss: Average training loss over the epoch.
            - accuracy: Standard accuracy over all training samples.
            - balanced_accuracy: Balanced accuracy (macro recall) over
              all training samples.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        patches = batch["patch"].to(device)
        coords = batch["coords"].to(device)
        modality = batch["modality"].to(device)
        # labels = batch["y"].to(device)  # single integer class label tensor (B,)
        locations = batch["location"].to(device)

        optimizer.zero_grad()
        outputs = model(patches, coords, modality)  # logits (B, num_classes)
        loss = criterion(outputs, locations.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(locations.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, balanced_accuracy


def eval(model, dataloader, criterion, device):
    """
    Run evaluation on dataloader for the patch classifier.

    Args:
        model (torch.nn.Module): Patch-level classification model in evaluation
            mode. Expected forward signature: ``model(patches, coords, modality)``
            returning logits of shape (B, num_classes).
        dataloader (torch.utils.data.DataLoader): Dataloader yielding batches
            of dictionaries with keys:
                - "patch": image or volume patches of shape (B, C, H, W, ...)
                - "coords": coordinate tensor of shape (B, D_coord)
                - "modality": modality encoding tensor of shape (B, D_mod)
                - "y": integer class labels of shape (B,)
        criterion (callable): Loss function taking (logits, targets) and
            returning a scalar loss.
        device (torch.device): Device on which computations are performed.

    Returns:
        tuple[float, float, float]:
            - avg_loss: Average evaluation loss over the epoch.
            - accuracy: Standard accuracy over all evaluation samples.
            - balanced_accuracy: Balanced accuracy (macro recall) over
              all evaluation samples.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            patches = batch["patch"].to(device)
            coords = batch["coords"].to(device)
            modality = batch["modality"].to(device)
            locations = batch["location"].to(device)

            outputs = model(patches, coords, modality)
            loss = criterion(outputs, locations.long())
            running_loss += loss.item()

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(locations.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, balanced_accuracy


def train_model(
    model,
    device,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    patience,
    output_dir,
):
    """
    Train the patch classifier model with early stopping and checkpointing.

    Runs a training loop over multiple epochs, evaluating on a validation set
    after each epoch. Tracks the best validation accuracy, performs early
    stopping based on lack of improvement, and saves model weights and
    training metrics to disk whenever a new best model is found.

    Args:
        model (torch.nn.Module): Patch-level classification model to train.
        device (torch.device): Device on which computations are performed.
        train_loader (torch.utils.data.DataLoader): Dataloader for the
            training set.
        val_loader (torch.utils.data.DataLoader): Dataloader for the
            validation set.
        criterion (callable): Loss function taking (logits, targets) and
            returning a scalar loss.
        optimizer (torch.optim.Optimizer): Optimizer used to update model
            parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler or
                   torch.optim.lr_scheduler.ReduceLROnPlateau): Learning-rate
            scheduler. Expected to be called as ``scheduler.step(val_loss)``.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of consecutive epochs without improvement in
            validation accuracy before triggering early stopping.
        output_dir (pathlib.Path): Directory where model weights and metrics
            CSV will be saved. The best weights are stored as
            ``output_dir / "weights.pth"`` and metrics as
            ``output_dir / "training_metrics.csv"``.

    Returns:
        tuple[dict, list[dict]]:
            - best_model_wts: State dict corresponding to the best validation
              accuracy encountered during training.
            - metrics: List of metric dictionaries, one per epoch, containing:
                - "epoch"
                - "train_loss", "train_acc", "train_balanced_acc"
                - "val_loss", "val_acc", "val_balanced_acc"
    """
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_since_best = 0
    metrics = []

    for epoch in range(epochs):
        train_loss, train_acc, train_bal_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_bal_acc = eval(model, val_loader, criterion, device)

        metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_balanced_acc": train_bal_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_balanced_acc": val_bal_acc,
            }
        )

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Train Balanced Acc: {train_bal_acc:.4f}"
        )
        print(
            f"Val   Loss: {val_loss:.4f} | Val   Accuracy: {val_acc:.4f} | Val  Balanced Acc: {val_bal_acc:.4f}"
        )

        # save trainign metrics to csv
        pd.DataFrame(metrics).to_csv(output_dir / "training_metrics.csv", index=False)

        # Check for improvement and save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # better save path
            torch.save(model.state_dict(), output_dir / "weights.pth")
            print("model saved.")
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Early stopping
        if epochs_since_best >= patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break

    return best_model_wts, metrics
