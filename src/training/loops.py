import copy

import pandas as pd
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None

from .metrics import regression_metrics


def predict_df(model, loader, device, cols=None):
    frames = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            frame = pd.DataFrame(
                {
                    "pred_norm": model(batch).view(-1).cpu().numpy(),
                    "actual_norm": batch.y.view(-1).cpu().numpy(),
                }
            )

            for col in cols or []:
                value = getattr(batch, col)
                if torch.is_tensor(value):
                    frame[col] = value.view(-1).cpu().numpy()
                else:
                    frame[col] = list(value)

            frames.append(frame)

    return pd.concat(frames, ignore_index=True)


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


def _group_metrics_from_dataframe(df, group_cols, loss_fn):
    group_metrics = {}
    for col in group_cols:
        if col not in df.columns:
            raise KeyError(f"Grouping column {col!r} not found in predictions dataframe.")

        column_metrics = {}
        for group_value, group_df in df.groupby(col, dropna=False):
            predictions = torch.tensor(group_df["pred_norm"].values, dtype=torch.float32)
            targets = torch.tensor(group_df["actual_norm"].values, dtype=torch.float32)

            loss = loss_fn(predictions, targets).item()
            rmse, mae = regression_metrics(predictions, targets)
            column_metrics[group_value] = {
                "loss": loss,
                "rmse": rmse,
                "mae": mae,
                "n": int(len(group_df)),
            }

        group_metrics[col] = column_metrics

    return group_metrics


def evaluate_by_groups(model, loader, loss_fn, device, group_cols):
    results_df = predict_df(model, loader, device, cols=group_cols)
    avg_loss, rmse, mae = evaluate(model, loader, loss_fn, device)
    return avg_loss, rmse, mae, _group_metrics_from_dataframe(results_df, group_cols, loss_fn)


def _init_history(record_categories=None, include_val=False, include_test=False):

    history = {"history_all": {}}
    if record_categories is not None:
        for category in record_categories:
            history[f"history_{category}"] = {f"history_{category}_group": {}}

    for category_name, category_history in history.items():
        category_history["train_loss"] = []
        if include_val:
            category_history.update({"val_loss": [], "val_rmse": [], "val_mae": []})
        if include_test:
            category_history.update({"test_loss": [], "test_rmse": [], "test_mae": []})

        category_history.update(
            {
                "best_epoch": None,
                "best_monitor_value": None,
                "monitor_name": None,
                "stopped_early": False,
                "epochs_ran": 0,
            }
        )

    return history


def _record_categories(history, prefix, loss, rmse, mae):
    history[f"{prefix}_loss"].append(loss)
    history[f"{prefix}_rmse"].append(rmse)
    history[f"{prefix}_mae"].append(mae)


def _record_group_metrics(history, group_key, prefix, group_metrics):
    group_history = history.setdefault(group_key, {})
    epoch_index = len(history["train_loss"]) - 1
    all_keys = [
        "train_loss",
        "train_rmse",
        "train_mae",
        "val_loss",
        "val_rmse",
        "val_mae",
        "test_loss",
        "test_rmse",
        "test_mae",
    ]

    if not group_metrics:
        return

    for group_value, value_metrics in group_metrics.items():
        entry = group_history.setdefault(
            group_value,
            {
                **{key: [] for key in all_keys},
                "train_n": None,
                "val_n": None,
                "test_n": None,
            },
        )

        while len(entry["train_loss"]) < epoch_index:
            for key in all_keys:
                entry[key].append(None)

        entry[f"{prefix}_loss"].append(value_metrics["loss"])
        entry[f"{prefix}_rmse"].append(value_metrics["rmse"])
        entry[f"{prefix}_mae"].append(value_metrics["mae"])
        entry[f"{prefix}_n"] = value_metrics.get("n")


def _format_progress(prefix, loss, rmse=None, mae=None):
    metrics = [f"{prefix} Loss = {loss:.4f}"]
    if rmse is not None:
        metrics.append(f"{prefix} RMSE = {rmse:.4f}")
    if mae is not None:
        metrics.append(f"{prefix} MAE = {mae:.4f}")
    return ", ".join(metrics)


def _build_progress_postfix(train_loss, val_metrics=None, test_metrics=None, optimizer=None):
    postfix = {"train_loss": f"{train_loss:.4f}"}

    if val_metrics is not None:
        postfix.update(
            {
                "val_loss": f"{val_metrics[0]:.4f}",
                "val_rmse": f"{val_metrics[1]:.4f}",
                "val_mae": f"{val_metrics[2]:.4f}",
            }
        )

    if test_metrics is not None:
        postfix.update(
            {
                "test_loss": f"{test_metrics[0]:.4f}",
                "test_rmse": f"{test_metrics[1]:.4f}",
                "test_mae": f"{test_metrics[2]:.4f}",
            }
        )

    if optimizer is not None and optimizer.param_groups:
        postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"

    return postfix


def _current_lr(optimizer):
    if optimizer is None or not optimizer.param_groups:
        return None
    return optimizer.param_groups[0].get("lr")


def _build_run_log(epoch, train_loss, val_metrics=None, test_metrics=None, optimizer=None):
    metrics = {
        "epoch": epoch + 1,
        "train/loss": train_loss,
    }

    if val_metrics is not None:
        metrics.update(
            {
                "val/loss": val_metrics[0],
                "val/rmse": val_metrics[1],
                "val/mae": val_metrics[2],
            }
        )

    if test_metrics is not None:
        metrics.update(
            {
                "test/loss": test_metrics[0],
                "test/rmse": test_metrics[1],
                "test/mae": test_metrics[2],
            }
        )

    lr = _current_lr(optimizer)
    if lr is not None:
        metrics["optimizer/lr"] = lr

    return metrics


def _finalize_group_history(history, group_key):
    if group_key not in history:
        return

    epoch_len = len(history["train_loss"])
    all_keys = [
        "train_loss",
        "train_rmse",
        "train_mae",
        "val_loss",
        "val_rmse",
        "val_mae",
        "test_loss",
        "test_rmse",
        "test_mae",
    ]

    for entry in history[group_key].values():
        for key in all_keys:
            while len(entry[key]) < epoch_len:
                entry[key].append(None)


def _propagate_history_metadata(history):
    metadata_keys = (
        "best_epoch",
        "best_monitor_value",
        "monitor_name",
        "stopped_early",
        "epochs_ran",
    )
    metadata = {key: history["history_all"].get(key) for key in metadata_keys}

    for history_name, category_history in history.items():
        if history_name == "history_all":
            continue

        category_history.update(metadata)
        group_key = f"{history_name}_group"
        for group_history in category_history.get(group_key, {}).values():
            group_history.update(metadata)


def _write_progress_message(progress_bar, message):
    if progress_bar is not None:
        progress_bar.write(message)
        return

    print(message)


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
    record_categories=None,
    run=None,
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
    history = _init_history(record_categories=record_categories, include_val=val_loader is not None, include_test=test_loader is not None)
    for category in history:
        history[category]["monitor_name"] = monitor_name

    best_monitor_value = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    progress_bar = None
    epoch_iterator = range(epochs)

    if verbose_every and tqdm is not None:
        progress_bar = tqdm(epoch_iterator, total=epochs, dynamic_ncols=True, unit="epoch")
        epoch_iterator = progress_bar

    try:
        for epoch in epoch_iterator:
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            history["history_all"]["train_loss"].append(train_loss)

            train_group_metrics = None
            if record_categories is not None:
                _, _, _, train_group_metrics = evaluate_by_groups(
                    model, train_loader, loss_fn, device, record_categories
                )

            val_metrics = None
            val_group_metrics = None
            if val_loader is not None:
                if record_categories is not None:
                    val_loss, val_rmse, val_mae, val_group_metrics = evaluate_by_groups(
                        model, val_loader, loss_fn, device, record_categories
                    )
                    val_metrics = (val_loss, val_rmse, val_mae)
                else:
                    val_metrics = evaluate(model, val_loader, loss_fn, device)

            test_metrics = None
            test_group_metrics = None
            if test_loader is not None:
                if record_categories is not None:
                    test_loss, test_rmse, test_mae, test_group_metrics = evaluate_by_groups(
                        model, test_loader, loss_fn, device, record_categories
                    )
                    test_metrics = (test_loss, test_rmse, test_mae)
                else:
                    test_metrics = evaluate(model, test_loader, loss_fn, device)

            if val_metrics is not None:
                history["history_all"]["val_loss"].append(val_metrics[0])
                history["history_all"]["val_rmse"].append(val_metrics[1])
                history["history_all"]["val_mae"].append(val_metrics[2])

            if test_metrics is not None:
                history["history_all"]["test_loss"].append(test_metrics[0])
                history["history_all"]["test_rmse"].append(test_metrics[1])
                history["history_all"]["test_mae"].append(test_metrics[2])

            for metric_name, metric_history in history.items():
                if metric_name == "history_all":
                    continue

                metric_history["train_loss"].append(train_loss)
                if val_metrics is not None:
                    _record_categories(metric_history, "val", *val_metrics)
                if test_metrics is not None:
                    _record_categories(metric_history, "test", *test_metrics)

                if record_categories is not None:
                    category = metric_name.replace("history_", "", 1)
                    group_key = f"{metric_name}_group"
                    if train_group_metrics is not None:
                        _record_group_metrics(metric_history, group_key, "train", train_group_metrics.get(category, {}))
                    if val_group_metrics is not None:
                        _record_group_metrics(metric_history, group_key, "val", val_group_metrics.get(category, {}))
                    if test_group_metrics is not None:
                        _record_group_metrics(metric_history, group_key, "test", test_group_metrics.get(category, {}))
                    _finalize_group_history(metric_history, group_key)

            if val_metrics is not None:
                monitor_value = val_metrics[0]
            elif test_metrics is not None:
                monitor_value = test_metrics[0]
            else:
                monitor_value = train_loss

            if scheduler is not None:
                scheduler.step(monitor_value)

            if progress_bar is not None:
                progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}")
                progress_bar.set_postfix(
                    _build_progress_postfix(train_loss, val_metrics, test_metrics, optimizer)
                )
            elif verbose_every and (epoch == 0 or (epoch + 1) % verbose_every == 0):
                progress_parts = [_format_progress("Train", train_loss)]
                if val_metrics is not None:
                    progress_parts.append(_format_progress("Val", *val_metrics))
                if test_metrics is not None:
                    progress_parts.append(_format_progress("Test", *test_metrics))
                print(f"Epoch {epoch + 1}: " + ", ".join(progress_parts))

            if run is not None:
                run.log(
                    _build_run_log(
                        epoch,
                        train_loss,
                        val_metrics=val_metrics,
                        test_metrics=test_metrics,
                        optimizer=optimizer,
                    )
                )

            if monitor_value < best_monitor_value - early_stopping_min_delta:
                best_monitor_value = monitor_value
                best_model_state = copy.deepcopy(model.state_dict())
                history["history_all"]["best_epoch"] = epoch
                history["history_all"]["best_monitor_value"] = monitor_value
                epochs_without_improvement = 0
            elif early_stopping_patience is not None:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    history["history_all"]["stopped_early"] = True
                    _write_progress_message(
                        progress_bar,
                        f"Early stopping at epoch {epoch + 1}: "
                        f"no improvement in {monitor_name} for {early_stopping_patience} epochs.",
                    )
                    break
    finally:
        if progress_bar is not None:
            progress_bar.close()

    model.load_state_dict(best_model_state)
    history["history_all"]["epochs_ran"] = len(history["history_all"]["train_loss"])
    _propagate_history_metadata(history)

    if run is not None:
        run.summary.update(
            {
                "best_epoch": history["history_all"]["best_epoch"],
                "best_monitor_value": history["history_all"]["best_monitor_value"],
                "monitor_name": history["history_all"]["monitor_name"],
                "stopped_early": history["history_all"]["stopped_early"],
                "epochs_ran": history["history_all"]["epochs_ran"],
            }
        )

    return model, history
