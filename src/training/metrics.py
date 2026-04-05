import torch


def compute_rmse(predictions, targets):
    return torch.sqrt(((predictions - targets) ** 2).mean()).item()


def compute_mae(predictions, targets):
    return torch.abs(predictions - targets).mean().item()


def regression_metrics(predictions, targets):
    return compute_rmse(predictions, targets), compute_mae(predictions, targets)
