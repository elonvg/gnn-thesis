from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.data.graph_building import build_graph_features
from src.training.loops import predict_df
from src.visualization.result_plots import summarize_by_group


class LoaderBatch(SimpleNamespace):
    def to(self, device):
        return self


def test_build_graph_features_attaches_row_id():
    df = pd.DataFrame(
        {
            "features": [
                SimpleNamespace(x=torch.ones(2, 3)),
                SimpleNamespace(x=torch.ones(1, 3)),
            ],
            "log10c": [1.0, 2.0],
            "duration": [24.0, 48.0],
            "taxid": [101, 202],
        }
    )
    df_tax = pd.DataFrame({"species": [1, 2]})

    features = build_graph_features(df, df_tax, tax_embedding=None)

    assert features[0].row_id.item() == 0
    assert features[1].row_id.item() == 1


def test_predict_df_collects_tensor_and_list_fields():
    class OffsetModel(nn.Module):
        def forward(self, batch):
            return batch.y.unsqueeze(-1) + 0.5

    loader = [
        LoaderBatch(
            y=torch.tensor([1.0, 2.0]),
            row_id=torch.tensor([10, 11]),
            effect=torch.tensor([1, 2]),
            smiles=["A", "B"],
        ),
        LoaderBatch(
            y=torch.tensor([3.0]),
            row_id=torch.tensor([12]),
            effect=torch.tensor([3]),
            smiles=["C"],
        ),
    ]

    results_df = predict_df(
        OffsetModel(),
        loader,
        device="cpu",
        cols=["row_id", "effect", "smiles"],
    )

    assert results_df["pred_norm"].tolist() == [1.5, 2.5, 3.5]
    assert results_df["actual_norm"].tolist() == [1.0, 2.0, 3.0]
    assert results_df["row_id"].tolist() == [10, 11, 12]
    assert results_df["effect"].tolist() == [1, 2, 3]
    assert results_df["smiles"].tolist() == ["A", "B", "C"]


def test_summarize_by_group_uses_train_means_with_global_fallback():
    train_df = pd.DataFrame(
        {
            "species_group": ["fish", "fish", "algae", "algae"],
            "log10c": [1.0, 3.0, 10.0, 12.0],
        }
    )
    results_df = pd.DataFrame(
        {
            "species_group": ["fish", "algae", "mollusks"],
            "actual_log10c": [1.0, 11.0, 5.0],
            "pred_log10c": [1.2, 10.7, 4.5],
        }
    )

    summary = summarize_by_group(results_df, train_df, "species_group", min_count=1)

    fish_row = summary.loc[summary["group"] == "fish"].iloc[0]
    mollusk_row = summary.loc[summary["group"] == "mollusks"].iloc[0]

    assert abs(fish_row["baseline_mae"] - 1.0) < 1e-9
    assert abs(fish_row["model_mae"] - 0.2) < 1e-9
    assert abs(fish_row["mae_gain"] - 0.8) < 1e-9
    assert abs(mollusk_row["baseline_mae"] - 1.5) < 1e-9
    assert fish_row["train_n"] == 2
    assert fish_row["baseline_source"] == "train_group_mean"
    assert mollusk_row["train_n"] == 0
    assert mollusk_row["baseline_source"] == "global_train_mean"


def test_summarize_by_group_accepts_actual_log10c_training_column():
    train_df = pd.DataFrame(
        {
            "species_group": ["fish", "fish", "algae"],
            "actual_log10c": [1.0, 3.0, 10.0],
        }
    )
    results_df = pd.DataFrame(
        {
            "species_group": ["fish", "algae"],
            "actual_log10c": [2.0, 10.0],
            "pred_log10c": [2.5, 9.0],
        }
    )

    summary = summarize_by_group(results_df, train_df, "species_group", min_count=1)

    fish_row = summary.loc[summary["group"] == "fish"].iloc[0]

    assert fish_row["train_n"] == 2
    assert fish_row["baseline_source"] == "train_group_mean"
    assert abs(fish_row["baseline_log10c"] - 2.0) < 1e-9
    assert abs(fish_row["baseline_mae"] - 0.0) < 1e-9
    assert abs(fish_row["model_mae"] - 0.5) < 1e-9


def test_summarize_by_group_requires_train_target_column():
    train_df = pd.DataFrame({"species_group": ["fish", "algae"]})
    results_df = pd.DataFrame(
        {
            "species_group": ["fish"],
            "actual_log10c": [1.0],
            "pred_log10c": [1.2],
        }
    )

    with pytest.raises(KeyError, match="train_df must include one of the training target columns"):
        summarize_by_group(results_df, train_df, "species_group", min_count=1)
