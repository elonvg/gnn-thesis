import copy

import torch

from .metrics import regression_metrics


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch).squeeze()
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    predictions, targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze()
            loss = loss_fn(out, batch.y)
            total_loss += loss.item()

            predictions.append(out.cpu())
            targets.append(batch.y.cpu())

    avg_loss = total_loss / len(loader)
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    rmse, mae = regression_metrics(predictions, targets)

    return avg_loss, rmse, mae


def _init_history(include_val=False, include_test=False):
    history = {"train_loss": []}

    if include_val:
        history.update({"val_loss": [], "val_rmse": [], "val_mae": []})
    if include_test:
        history.update({"test_loss": [], "test_rmse": [], "test_mae": []})

    history.update(
        {
            "best_epoch": None,
            "best_monitor_value": None,
            "monitor_name": None,
            "stopped_early": False,
            "epochs_ran": 0,
        }
    )

    return history


def _record_metrics(history, prefix, loss, rmse, mae):
    history[f"{prefix}_loss"].append(loss)
    history[f"{prefix}_rmse"].append(rmse)
    history[f"{prefix}_mae"].append(mae)


def _format_progress(prefix, loss, rmse=None, mae=None):
    metrics = [f"{prefix} Loss = {loss:.4f}"]
    if rmse is not None:
        metrics.append(f"{prefix} RMSE = {rmse:.4f}")
    if mae is not None:
        metrics.append(f"{prefix} MAE = {mae:.4f}")
    return ", ".join(metrics)


def train(
    model,
    train_loader,
    test_loader=None,
    loss_fn=None,
    optimizer=None,
    scheduler=None,
    epochs=100,
    device="cpu",
    val_loader=None,
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
    verbose_every=10,
):
    model = model.to(device)

    if loss_fn is None or optimizer is None:
        raise ValueError("loss_fn and optimizer must both be provided.")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive when provided.")

    monitor_name = (
        "val_loss"
        if val_loader is not None
        else "test_loss"
        if test_loader is not None
        else "train_loss"
    )
    history = _init_history(include_val=val_loader is not None, include_test=test_loader is not None)
    history["monitor_name"] = monitor_name

    best_monitor_value = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        history["train_loss"].append(train_loss)

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, loss_fn, device)
            _record_metrics(history, "val", *val_metrics)

        test_metrics = None
        if test_loader is not None:
            test_metrics = evaluate(model, test_loader, loss_fn, device)
            _record_metrics(history, "test", *test_metrics)

        if val_metrics is not None:
            monitor_value = val_metrics[0]
        elif test_metrics is not None:
            monitor_value = test_metrics[0]
        else:
            monitor_value = train_loss

        if scheduler is not None:
            scheduler.step(monitor_value)

        if verbose_every and epoch % verbose_every == 0:
            progress_parts = [_format_progress("Train", train_loss)]
            if val_metrics is not None:
                progress_parts.append(_format_progress("Val", *val_metrics))
            if test_metrics is not None:
                progress_parts.append(_format_progress("Test", *test_metrics))
            print(f"Epoch {epoch}: " + ", ".join(progress_parts))

        if monitor_value < best_monitor_value - early_stopping_min_delta:
            best_monitor_value = monitor_value
            best_model_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            history["best_monitor_value"] = monitor_value
            epochs_without_improvement = 0
        elif early_stopping_patience is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                history["stopped_early"] = True
                print(
                    f"Early stopping at epoch {epoch}: "
                    f"no improvement in {monitor_name} for {early_stopping_patience} epochs."
                )
                break

    model.load_state_dict(best_model_state)
    history["epochs_ran"] = len(history["train_loss"])

    return model, history
