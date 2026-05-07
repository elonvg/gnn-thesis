import torch


def compute_rmse(predictions, targets):
    return torch.sqrt(((predictions - targets) ** 2).mean()).item()


def compute_mean_ae(predictions, targets):
    return torch.abs(predictions - targets).mean().item()


def compute_median_ae(predictions, targets):
    return torch.abs(predictions - targets).median().item()


def regression_metrics(predictions, targets):
    return compute_rmse(predictions, targets), compute_mean_ae(predictions, targets), compute_median_ae(predictions, targets)
