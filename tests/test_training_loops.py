from types import SimpleNamespace

import torch
import torch.nn as nn

from src.training.loops import evaluate_by_groups, train


class LoaderBatch(SimpleNamespace):
    def to(self, device):
        return self


class IdentityRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.linear.weight.data.fill_(1.0)

    def forward(self, batch):
        return self.linear(batch.y.unsqueeze(-1))


def test_evaluate_by_groups_returns_group_metrics():
    loader = [
        LoaderBatch(y=torch.tensor([1.0, 3.0]), species_group=["fish", "algae"]),
    ]
    model = IdentityRegressionModel()
    loss_fn = nn.MSELoss()

    avg_loss, rmse, mae, group_metrics = evaluate_by_groups(
        model, loader, loss_fn, device="cpu", group_cols=["species_group"]
    )

    assert avg_loss == 0.0
    assert rmse == 0.0
    assert mae == 0.0
    assert "species_group" in group_metrics
    assert group_metrics["species_group"]["fish"]["loss"] == 0.0
    assert group_metrics["species_group"]["algae"]["mae"] == 0.0


def test_train_records_group_history_for_record_categories():
    loader = [
        LoaderBatch(y=torch.tensor([1.0, 2.0]), species_group=["fish", "fish"]),
    ]
    model = IdentityRegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    _, history = train(
        model,
        train_loader=loader,
        val_loader=loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=1,
        device="cpu",
        record_categories=["species_group"],
    )

    assert "history_all" in history
    assert "history_species_group" in history
    assert history["history_species_group"]["train_loss"] == [0.0]
    assert history["history_species_group"]["val_loss"] == [0.0]
    assert history["history_species_group"]["history_species_group_group"]["fish"]["val_loss"] == [0.0]
    assert history["history_species_group"]["history_species_group_group"]["fish"]["train_loss"] == [0.0]
    assert history["history_species_group"]["history_species_group_group"]["fish"]["train_n"] == 2
    assert history["history_species_group"]["history_species_group_group"]["fish"]["val_n"] == 2
