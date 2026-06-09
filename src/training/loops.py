import copy
from contextlib import nullcontext

import pandas as pd
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None

from .metrics import regression_metrics


METRIC_NAMES = ("loss", "rmse", "mean_ae", "median_ae")
SPLIT_PREFIXES = ("train", "val", "test")


def _amp_dtype_from_name(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype

    dtype_name = str(dtype).lower().replace("torch.", "")
    if dtype_name in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16

    raise ValueError(
        "amp_dtype must be one of: 'float16', 'fp16', 'bfloat16', 'bf16'."
    )


def _autocast_context(device, enabled=False, dtype=torch.float16):
    if not enabled:
        return nullcontext()

    device_type = torch.device(device).type
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device_type, dtype=dtype)
        except TypeError:
            return torch.amp.autocast(device_type, dtype=dtype)

    return torch.cuda.amp.autocast(dtype=dtype)


def _make_grad_scaler(enabled):
    if not enabled:
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device="cuda", enabled=True)
        except TypeError:
            pass

    return torch.cuda.amp.GradScaler(enabled=True)


def predict_df(model, loader, device, cols=None):
    return _collect_eval_outputs(model, loader, None, device, cols=cols)["results_df"]


def _as_batch_values(value, batch_size):
    if torch.is_tensor(value):
        return value.detach().cpu().view(-1).tolist()

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()

    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        value = [value]

    if len(value) != batch_size:
        raise ValueError(
            f"Expected {batch_size} values for a batched attribute, got {len(value)}."
        )

    return list(value)


def _collect_eval_outputs(model, loader, loss_fn, device, cols=None):
    model.eval()
    cols = list(dict.fromkeys(cols or []))
    total_loss = 0
    predictions, targets = [], []
    column_values = {col: [] for col in cols}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1)
            target = batch.y.view(-1)

            if loss_fn is not None:
                total_loss += loss_fn(out, target).item()

            batch_size = out.numel()
            predictions.append(out.detach().cpu())
            targets.append(target.detach().cpu())

            for col in cols:
                column_values[col].extend(_as_batch_values(getattr(batch, col), batch_size))

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    results = {
        "predictions": predictions,
        "targets": targets,
        "results_df": pd.DataFrame(
            {
                "pred_norm": predictions.numpy(),
                "actual_norm": targets.numpy(),
                **column_values,
            }
        ),
    }

    if loss_fn is not None:
        results["avg_loss"] = total_loss / len(loader)
        results["metrics"] = (results["avg_loss"], *regression_metrics(predictions, targets))

    return results


def train_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    mixed_precision=False,
    amp_dtype=torch.float16,
    scaler=None,
):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, enabled=mixed_precision, dtype=amp_dtype):
            out = model(batch).view(-1)
            loss = loss_fn(out, batch.y.view(-1))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, loss_fn, device):
    return _collect_eval_outputs(model, loader, loss_fn, device)["metrics"]


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
            rmse, mean_ae, median_ae = regression_metrics(predictions, targets)
            column_metrics[group_value] = {
                "loss": loss,
                "rmse": rmse,
                "mean_ae": mean_ae,
                "median_ae": median_ae,
                "n": int(len(group_df)),
            }

        group_metrics[col] = column_metrics

    return group_metrics


def evaluate_by_groups(model, loader, loss_fn, device, group_cols):
    eval_outputs = _evaluate_with_recorded_groups(
        model,
        loader,
        loss_fn,
        device,
        record_categories=group_cols,
    )
    return (*eval_outputs["metrics"], eval_outputs["group_metrics"])


def add_joint_metrics_to_history(
    history,
    joint_metrics,
    main_category,
    sub_category,
    prefix=None,
    epoch_index=None,
):
    joint_history = history.setdefault("joint_metrics", {}).setdefault(main_category, {})

    for main_value, sub_metrics in joint_metrics.items():
        main_history = joint_history.setdefault(main_value, {}).setdefault(sub_category, {})

        for sub_value, metrics in sub_metrics.items():
            entry = main_history.setdefault(sub_value, {})

            if prefix is None:
                entry.update(metrics)
                continue

            for metric_name, metric_value in metrics.items():
                key = f"{prefix}_{metric_name}"
                if metric_name == "n":
                    entry[key] = metric_value
                else:
                    values = entry.setdefault(key, [])
                    if epoch_index is not None:
                        while len(values) < epoch_index:
                            values.append(None)
                    values.append(metric_value)

    return history


def evaluate_by_joint_group(
    model,
    loader,
    loss_fn,
    device,
    main_category,
    sub_category,
    history=None,
    prefix=None,
):
    if main_category == sub_category:
        raise ValueError("main_category and sub_category must be different columns.")

    results_df = _collect_eval_outputs(
        model,
        loader,
        None,
        device,
        cols=[main_category, sub_category],
    )["results_df"]
    joint_metrics = _joint_metrics_from_dataframe(results_df, main_category, sub_category, loss_fn)

    if history is not None:
        add_joint_metrics_to_history(
            history,
            joint_metrics,
            main_category,
            sub_category,
            prefix=prefix,
        )

    return joint_metrics


def _joint_metrics_from_dataframe(df, main_category, sub_category, loss_fn):
    joint_metrics = {}

    for (main_value, sub_value), group_df in df.groupby(
        [main_category, sub_category], dropna=False
    ):
        predictions = torch.tensor(group_df["pred_norm"].values, dtype=torch.float32)
        targets = torch.tensor(group_df["actual_norm"].values, dtype=torch.float32)

        loss = loss_fn(predictions, targets).item()
        rmse, mean_ae, median_ae = regression_metrics(predictions, targets)

        main_metrics = joint_metrics.setdefault(main_value, {})
        main_metrics[sub_value] = {
            "loss": loss,
            "rmse": rmse,
            "mean_ae": mean_ae,
            "median_ae": median_ae,
            "n": int(len(group_df)),
        }

    return joint_metrics


def _evaluation_columns(record_categories=None, record_joint_categories=None):
    columns = []
    if record_categories is not None:
        columns.extend(record_categories)
    if record_joint_categories is not None:
        columns.extend(record_joint_categories)
    return list(dict.fromkeys(columns))


def _evaluate_with_recorded_groups(
    model,
    loader,
    loss_fn,
    device,
    record_categories=None,
    record_joint_categories=None,
):
    eval_outputs = _collect_eval_outputs(
        model,
        loader,
        loss_fn,
        device,
        cols=_evaluation_columns(record_categories, record_joint_categories),
    )
    results_df = eval_outputs["results_df"]
    group_metrics = (
        _group_metrics_from_dataframe(results_df, record_categories, loss_fn)
        if record_categories is not None
        else None
    )
    joint_metrics = None
    if record_joint_categories is not None:
        main_category, sub_category = record_joint_categories
        joint_metrics = _joint_metrics_from_dataframe(
            results_df,
            main_category,
            sub_category,
            loss_fn,
        )

    return {
        "metrics": eval_outputs["metrics"],
        "group_metrics": group_metrics,
        "joint_metrics": joint_metrics,
    }

def build_label_decoders(label_encoder):
    if label_encoder is None:
        return {}

    return {
        category: {encoded: original for original, encoded in encoder.items()}
        for category, encoder in label_encoder.items()
    }

def _init_history(record_categories=None, include_val=False, include_test=False):

    history = {"history_all": {}}
    if record_categories is not None:
        for category in record_categories:
            history[f"history_{category}"] = {f"history_{category}_group": {}}

    for category_name, category_history in history.items():
        category_history["train_loss"] = []
        if include_val:
            category_history.update({"val_loss": [], "val_rmse": [], "val_mean_ae": [], "val_median_ae": []})
        if include_test:
            category_history.update({"test_loss": [], "test_rmse": [], "test_mean_ae": [], "test_median_ae": []})

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


def _metric_history_keys():
    return [f"{prefix}_{metric}" for prefix in SPLIT_PREFIXES for metric in METRIC_NAMES]


def _record_split_metrics(history, prefix, metrics):
    if f"{prefix}_loss" not in history:
        return

    if metrics is None:
        for metric_name in METRIC_NAMES:
            history[f"{prefix}_{metric_name}"].append(None)
        return

    for metric_name, metric_value in zip(METRIC_NAMES, metrics):
        history[f"{prefix}_{metric_name}"].append(metric_value)


def _record_group_metrics(history, group_key, prefix, group_metrics):
    group_history = history.setdefault(group_key, {})
    epoch_index = len(history["train_loss"]) - 1
    all_keys = _metric_history_keys()

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
        entry[f"{prefix}_mean_ae"].append(value_metrics["mean_ae"])
        entry[f"{prefix}_median_ae"].append(value_metrics["median_ae"])
        entry[f"{prefix}_n"] = value_metrics.get("n")


def _format_progress(prefix, loss, rmse=None, mean_ae=None, median_ae=None):
    metrics = [f"{prefix} Loss = {loss:.4f}"]
    if rmse is not None:
        metrics.append(f"{prefix} RMSE = {rmse:.4f}")
    if mean_ae is not None:
        metrics.append(f"{prefix} mean_ae = {mean_ae:.4f}")
    if median_ae is not None:
        metrics.append(f"{prefix} median_ae = {median_ae:.4f}")
    return ", ".join(metrics)


def _build_progress_postfix(train_loss, val_metrics=None, test_metrics=None, optimizer=None):
    postfix = {"train_loss": f"{train_loss:.4f}"}

    if val_metrics is not None:
        postfix.update(
            {
                "val_loss": f"{val_metrics[0]:.4f}",
                "val_rmse": f"{val_metrics[1]:.4f}",
                "val_mean_ae": f"{val_metrics[2]:.4f}",
                "val_median_ae": f"{val_metrics[3]:.4f}",
            }
        )

    if test_metrics is not None:
        postfix.update(
            {
                "test_loss": f"{test_metrics[0]:.4f}",
                "test_rmse": f"{test_metrics[1]:.4f}",
                "test_mean_ae": f"{test_metrics[2]:.4f}",
                "test_median_ae": f"{test_metrics[3]:.4f}",
            }
        )

    if optimizer is not None and optimizer.param_groups:
        postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"

    return postfix


def _current_lr(optimizer):
    if optimizer is None or not optimizer.param_groups:
        return None
    return optimizer.param_groups[0].get("lr")


def _normalize_joint_categories(record_joint_categories):
    if record_joint_categories is None:
        return None
    if isinstance(record_joint_categories, str) or len(record_joint_categories) != 2:
        raise ValueError("record_joint_categories must be a pair of category names.")

    main_category, sub_category = record_joint_categories
    if main_category == sub_category:
        raise ValueError("record_joint_categories must contain two different categories.")

    return main_category, sub_category


def _build_run_log(
    epoch,
    history,
    train_loss,
    val_metrics=None,
    test_metrics=None,
    record_categories=None,
    record_joint_categories=None,
    label_decoder=None,
    optimizer=None,
):
    label_decoder = label_decoder or {}
    metrics = {
        "epoch": epoch + 1,
        "train/loss": history["history_all"]["train_loss"][-1],
    }

    if val_metrics is not None:
        metrics.update(
            {
                "val/loss": history["history_all"]["val_loss"][-1] if "val_loss" in history["history_all"] else None,
                "val/rmse": history["history_all"]["val_rmse"][-1] if "val_rmse" in history["history_all"] else None,
                "val/mean_ae": history["history_all"]["val_mean_ae"][-1] if "val_mean_ae" in history["history_all"] else None,
                "val/median_ae": history["history_all"]["val_median_ae"][-1] if "val_median_ae" in history["history_all"] else None,
            }
        )

    if test_metrics is not None:
        metrics.update(
            {
                "test/loss": history["history_all"]["test_loss"][-1] if "test_loss" in history["history_all"] else None,
                "test/rmse": history["history_all"]["test_rmse"][-1] if "test_rmse" in history["history_all"] else None,
                "test/mean_ae": history["history_all"]["test_mean_ae"][-1] if "test_mean_ae" in history["history_all"] else None,
                "test/median_ae": history["history_all"]["test_median_ae"][-1] if "test_median_ae" in history["history_all"] else None,
            }
        )
                
    if record_categories is not None:
        for category in record_categories:
            for group_value in history[f"history_{category}"].get(f"history_{category}_group", {}):
                label = str(label_decoder.get(category, {}).get(group_value, group_value))
                
                metrics.update(
                    {   
                        f"cat_{category}/{label}/val_rmse": history[f"history_{category}"][f"history_{category}_group"][group_value]["val_rmse"][-1],
                        f"cat_{category}/{label}/val_mean_ae": history[f"history_{category}"][f"history_{category}_group"][group_value]["val_mean_ae"][-1],
                        f"cat_{category}/{label}/val_median_ae": history[f"history_{category}"][f"history_{category}_group"][group_value]["val_median_ae"][-1],
                    }
            
        )

    if record_joint_categories is not None:
        main_category, sub_category = record_joint_categories
        joint_history = history.get("joint_metrics", {}).get(main_category, {})

        for main_value, main_history in joint_history.items():
            main_label = str(label_decoder.get(main_category, {}).get(main_value, main_value))

            for sub_value, sub_history in main_history.get(sub_category, {}).items():
                sub_label = str(label_decoder.get(sub_category, {}).get(sub_value, sub_value))
                metric_prefix = f"joint_{main_label}/{sub_category}/{sub_label}"

                if val_metrics is not None and sub_history.get("val_loss"):
                    metrics[f"{metric_prefix}/val_median_ae"] = sub_history["val_median_ae"][-1]

    lr = _current_lr(optimizer)
    if lr is not None:
        metrics["optimizer/lr"] = lr

    return metrics


def _finalize_group_history(history, group_key):
    if group_key not in history:
        return

    epoch_len = len(history["train_loss"])
    all_keys = _metric_history_keys()

    for entry in history[group_key].values():
        for key in all_keys:
            while len(entry[key]) < epoch_len:
                entry[key].append(None)


def _finalize_joint_history(history, main_category, sub_category):
    epoch_len = len(history["history_all"]["train_loss"])
    joint_history = history.get("joint_metrics", {}).get(main_category, {})
    all_keys = _metric_history_keys()

    for main_history in joint_history.values():
        for entry in main_history.get(sub_category, {}).values():
            for key in all_keys:
                values = entry.setdefault(key, [])
                while len(values) < epoch_len:
                    values.append(None)


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
        if history_name in ("history_all", "joint_metrics"):
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


def _validate_interval(name, value, allow_none=False, allow_zero=False):
    if value is None and allow_none:
        return value
    if allow_zero and value == 0:
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _should_run_interval(epoch, total_epochs, every):
    return (epoch + 1) % every == 0 or epoch == total_epochs - 1


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
    eval_every=1,
    log_every=None,
    record_categories=None,
    record_joint_categories=None,
    label_encoder=None,
    run=None,
    mixed_precision=False,
    amp_dtype="float16",
):
    device = torch.device(device)
    model = model.to(device)
    amp_dtype = _amp_dtype_from_name(amp_dtype)
    amp_enabled = bool(mixed_precision) and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled and amp_dtype == torch.float16)

    if loss_fn is None or optimizer is None:
        raise ValueError("loss_fn and optimizer must both be provided.")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive when provided.")
    eval_every = _validate_interval("eval_every", eval_every)
    log_every = _validate_interval(
        "log_every",
        log_every,
        allow_none=True,
        allow_zero=True,
    )
    if log_every is None:
        log_every = eval_every

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
    label_decoder = build_label_decoders(label_encoder) if label_encoder is not None else None
    record_joint_categories = _normalize_joint_categories(record_joint_categories)

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
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                mixed_precision=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
            history["history_all"]["train_loss"].append(train_loss)
            should_evaluate = _should_run_interval(epoch, epochs, eval_every)
            should_log = log_every != 0 and _should_run_interval(epoch, epochs, log_every)

            train_group_metrics = None
            train_joint_metrics = None
            record_detailed_metrics = record_categories is not None or record_joint_categories is not None
            if should_evaluate and record_detailed_metrics:
                train_eval_outputs = _evaluate_with_recorded_groups(
                    model,
                    train_loader,
                    loss_fn,
                    device,
                    record_categories=record_categories,
                    record_joint_categories=record_joint_categories,
                )
                train_group_metrics = train_eval_outputs["group_metrics"]
                train_joint_metrics = train_eval_outputs["joint_metrics"]

            val_metrics = None
            val_group_metrics = None
            val_joint_metrics = None
            if should_evaluate and val_loader is not None:
                if record_detailed_metrics:
                    val_eval_outputs = _evaluate_with_recorded_groups(
                        model,
                        val_loader,
                        loss_fn,
                        device,
                        record_categories=record_categories,
                        record_joint_categories=record_joint_categories,
                    )
                    val_metrics = val_eval_outputs["metrics"]
                    val_group_metrics = val_eval_outputs["group_metrics"]
                    val_joint_metrics = val_eval_outputs["joint_metrics"]
                else:
                    val_metrics = evaluate(model, val_loader, loss_fn, device)

            test_metrics = None
            test_group_metrics = None
            test_joint_metrics = None
            if should_evaluate and test_loader is not None:
                if record_detailed_metrics:
                    test_eval_outputs = _evaluate_with_recorded_groups(
                        model,
                        test_loader,
                        loss_fn,
                        device,
                        record_categories=record_categories,
                        record_joint_categories=record_joint_categories,
                    )
                    test_metrics = test_eval_outputs["metrics"]
                    test_group_metrics = test_eval_outputs["group_metrics"]
                    test_joint_metrics = test_eval_outputs["joint_metrics"]
                else:
                    test_metrics = evaluate(model, test_loader, loss_fn, device)

            _record_split_metrics(history["history_all"], "val", val_metrics)
            _record_split_metrics(history["history_all"], "test", test_metrics)

            for metric_name, metric_history in history.items():
                if metric_name in ("history_all", "joint_metrics"):
                    continue

                metric_history["train_loss"].append(train_loss)
                _record_split_metrics(metric_history, "val", val_metrics)
                _record_split_metrics(metric_history, "test", test_metrics)

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

            if record_joint_categories is not None:
                main_category, sub_category = record_joint_categories
                if train_joint_metrics is not None:
                    add_joint_metrics_to_history(
                        history,
                        train_joint_metrics,
                        main_category,
                        sub_category,
                        prefix="train",
                        epoch_index=epoch,
                    )
                if val_joint_metrics is not None:
                    add_joint_metrics_to_history(
                        history,
                        val_joint_metrics,
                        main_category,
                        sub_category,
                        prefix="val",
                        epoch_index=epoch,
                    )
                if test_joint_metrics is not None:
                    add_joint_metrics_to_history(
                        history,
                        test_joint_metrics,
                        main_category,
                        sub_category,
                        prefix="test",
                        epoch_index=epoch,
                    )
                _finalize_joint_history(history, main_category, sub_category)

            if val_metrics is not None:
                monitor_value = val_metrics[0]
            elif test_metrics is not None:
                monitor_value = test_metrics[0]
            else:
                monitor_value = train_loss if val_loader is None and test_loader is None else None

            if scheduler is not None and monitor_value is not None:
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

            if run is not None and should_log:
                run.log(
                    _build_run_log(
                        epoch,
                        history,
                        train_loss,
                        val_metrics=val_metrics,
                        test_metrics=test_metrics,
                        record_categories=record_categories,
                        record_joint_categories=record_joint_categories,
                        label_decoder=label_decoder,
                        optimizer=optimizer,
                    )
                )

            if monitor_value is None:
                continue

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
