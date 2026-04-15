from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.splitting import (
    _build_smiles_index_lookup,
    _take_split_indices,
    build_attribute_sampling_weights,
    build_weighted_random_sampler,
    butina_split,
    compute_attribute_distribution,
)
from src.models.meta_encoder import (
    CategoricalOneHot,
    MetaEncoder,
    NumericalEncoder,
    TaxonomyOneHot,
)
from src.models.taxonomic_embedder import PretrainedTaxidEncoder
from src.models.toxicity_model import ToxicityModel
from src.training.loops import train


def build_batch():
    return SimpleNamespace(
        duration=torch.tensor([24.0, 48.0]),
        temperature=torch.tensor([18.0, 21.5]),
        species_group=torch.tensor([1, 2], dtype=torch.long),
        family=torch.tensor([3, 1], dtype=torch.long),
        endpoint=torch.tensor([1, 2], dtype=torch.long),
        effect=torch.tensor([4, 3], dtype=torch.long),
    )


class LoaderBatch(SimpleNamespace):
    def to(self, device):
        return self


def test_numerical_encoder_single_column():
    batch = build_batch()
    encoder = NumericalEncoder(output_dim=12)

    output = encoder(batch)

    assert output.shape == (2, 12)


def test_numerical_encoder_multiple_columns():
    batch = build_batch()
    encoder = NumericalEncoder(
        numerical_columns=["duration", "temperature"],
        output_dim=12,
    )

    output = encoder(batch)

    assert output.shape == (2, 12)
    assert encoder.input_dim == 2


def test_taxonomy_one_hot_single_column():
    batch = build_batch()
    config = {"species_group": (4, 8)}
    encoder = TaxonomyOneHot(config, output_dim=16)

    output = encoder(batch)

    assert output.shape == (2, 16)


def test_taxonomy_one_hot_multiple_columns():
    batch = build_batch()
    config = {
        "species_group": (4, 8),
        "family": (5, 4),
    }
    encoder = TaxonomyOneHot(config, output_dim=16)

    output = encoder(batch)

    assert output.shape == (2, 16)
    assert encoder.raw_dim == 9


def test_meta_encoder_with_taxonomy_one_hot():
    batch = build_batch()
    config = {
        "species_group": (4, 8),
        "family": (5, 4),
    }
    encoder = MetaEncoder(
        config_tax=config,
        hidden_dim=12,
        output_dim=20,
        taxonomy_encoder_cls=TaxonomyOneHot,
    )

    output = encoder(batch)

    assert output.shape == (2, 32)


def test_categorical_one_hot_encoder_for_endpoint_and_effect():
    batch = build_batch()
    config = {
        "endpoint": 5,
        "effect": 6,
    }
    encoder = CategoricalOneHot(config, output_dim=10)

    output = encoder(batch)

    assert output.shape == (2, 10)
    assert encoder.raw_dim == 11


def test_meta_encoder_with_taxonomy_and_categorical_one_hot():
    batch = build_batch()
    taxonomy_config = {
        "species_group": (4, 8),
        "family": (5, 4),
    }
    categorical_config = {
        "endpoint": 5,
        "effect": 6,
    }
    encoder = MetaEncoder(
        config_tax=taxonomy_config,
        hidden_dim=12,
        output_dim=20,
        taxonomy_encoder_cls=TaxonomyOneHot,
        config_categorical=categorical_config,
        categorical_output_dim=10,
    )

    output = encoder(batch)

    assert output.shape == (2, 42)
    assert encoder.output_dim == 42


def test_meta_encoder_accepts_categorical_config_alias():
    batch = build_batch()
    encoder = MetaEncoder(
        config_tax={"species_group": (4, 8)},
        hidden_dim=12,
        output_dim=20,
        taxonomy_encoder_cls=TaxonomyOneHot,
        categorical_config={"endpoint": 5, "effect": 6},
        categorical_output_dim=10,
    )

    output = encoder(batch)

    assert output.shape == (2, 42)
    assert encoder.output_dim == 42


def test_meta_encoder_with_multiple_numerical_columns():
    batch = build_batch()
    encoder = MetaEncoder(
        meta_dim=2,
        hidden_dim=12,
        numerical_columns=["duration", "temperature"],
    )

    output = encoder(batch)

    assert output.shape == (2, 12)
    assert encoder.output_dim == 12


def test_pretrained_taxid_encoder_reads_raw_taxids():
    encoder = PretrainedTaxidEncoder(
        taxonomic_embedding_dict={
            "10090": np.ones(4, dtype=np.float32),
            "10116": np.full(4, 2.0, dtype=np.float32),
        },
        embedding_dim=4,
        output_dim=6,
        dropout=0.0,
    )
    batch = LoaderBatch(taxid_raw=torch.tensor([10090, 10116, 0], dtype=torch.long))

    output = encoder(batch)

    assert output.shape == (3, 6)


def test_meta_encoder_can_append_pretrained_taxid_embeddings():
    batch = LoaderBatch(
        duration=torch.tensor([24.0, 48.0]),
        taxid_raw=torch.tensor([10090, 10116], dtype=torch.long),
    )
    encoder = MetaEncoder(
        hidden_dim=12,
        numerical_columns=["duration"],
        pretrained_taxid_output_dim=8,
        pretrained_taxid_encoder_kwargs={
            "taxonomic_embedding_dict": {
                "10090": np.ones(4, dtype=np.float32),
                "10116": np.full(4, 2.0, dtype=np.float32),
            },
            "embedding_dim": 4,
            "dropout": 0.0,
        },
    )

    output = encoder(batch)

    assert output.shape == (2, 20)
    assert encoder.output_dim == 20


def test_take_split_indices_preserves_duplicate_rows():
    smiles_list = ["A", "B", "A", "C", "B"]
    indices_by_smile = _build_smiles_index_lookup(smiles_list)

    train_indices = _take_split_indices(["A", "B", "A"], indices_by_smile)
    test_indices = _take_split_indices(["C", "B"], indices_by_smile)

    assert train_indices == [0, 1, 2]
    assert test_indices == [3, 4]
    assert set(train_indices).isdisjoint(test_indices)


def test_compute_attribute_distribution_reads_scalar_tensor_attributes():
    dataset = [
        SimpleNamespace(species_group=torch.tensor(1, dtype=torch.long)),
        SimpleNamespace(species_group=torch.tensor(1, dtype=torch.long)),
        SimpleNamespace(species_group=torch.tensor(2, dtype=torch.long)),
    ]

    distribution = compute_attribute_distribution(dataset, "species_group")

    assert distribution == pytest.approx({1: 2 / 3, 2: 1 / 3})


def test_weighted_sampler_weights_match_reference_distribution_for_present_species():
    dataset = [
        SimpleNamespace(species_group=torch.tensor(1, dtype=torch.long)),
        SimpleNamespace(species_group=torch.tensor(1, dtype=torch.long)),
        SimpleNamespace(species_group=torch.tensor(1, dtype=torch.long)),
        SimpleNamespace(species_group=torch.tensor(2, dtype=torch.long)),
    ]

    weights = build_attribute_sampling_weights(
        dataset,
        "species_group",
        target_distribution={1: 0.4, 2: 0.4, 3: 0.2},
    )
    sampler = build_weighted_random_sampler(
        dataset,
        "species_group",
        target_distribution={1: 0.4, 2: 0.4, 3: 0.2},
    )

    assert weights.tolist() == pytest.approx([1 / 6, 1 / 6, 1 / 6, 1 / 2])
    assert sampler.weights.tolist() == pytest.approx([1 / 6, 1 / 6, 1 / 6, 1 / 2])
    assert sampler.num_samples == len(dataset)


def test_butina_split_returns_validation_split_without_reusing_duplicate_rows():
    features = [
        SimpleNamespace(smiles=smiles, y=torch.tensor(float(idx)), row_id=idx)
        for idx, smiles in enumerate(["A", "A", "B", "C", "B", "D"])
    ]

    def fake_butina_train_test_split(smiles_list, y_list, train_size, test_size):
        return (
            smiles_list[:train_size],
            smiles_list[train_size:train_size + test_size],
            y_list[:train_size],
            y_list[train_size:train_size + test_size],
        )

    with patch("src.data.splitting.butina_train_test_split", side_effect=fake_butina_train_test_split):
        train_dataset, test_dataset, val_dataset = butina_split(
            features,
            frac_train=0.5,
            frac_test=0.25,
            frac_valid=0.25,
        )

    assert [item.row_id for item in train_dataset] == [0, 1]
    assert [item.row_id for item in val_dataset] == [2, 3]
    assert [item.row_id for item in test_dataset] == [4, 5]


def test_butina_split_can_use_precomputed_cluster_csv():
    features = [
        SimpleNamespace(smiles=smiles, y=torch.tensor(float(idx)), row_id=idx)
        for idx, smiles in enumerate(["A", "A", "B", "C", "B", "D"])
    ]

    csv_content = "\n".join(
        [
            "SMILES,Cluster_at_cutoff_0.3",
            "A,10",
            "B,20",
            "C,20",
            "D,30",
        ]
    )

    with TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "butina_clusters.csv"
        csv_path.write_text(csv_content)

        train_dataset, test_dataset, val_dataset = butina_split(
            features,
            frac_train=0.5,
            frac_test=0.25,
            frac_valid=0.25,
            cluster_csv_path=csv_path,
            cluster_column="Cluster_at_cutoff_0.3",
        )

    assert [item.row_id for item in train_dataset] == [2, 3, 4]
    assert [item.row_id for item in val_dataset] == [0, 1]
    assert [item.row_id for item in test_dataset] == [5]


def test_toxicity_model_infers_encoder_dimensions():
    class DummyGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_dim = 5

        def forward(self, data):
            return data.mol_embed

    class DummyMetaEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_dim = 3

        def forward(self, data):
            return data.meta_embed

    batch = LoaderBatch(
        mol_embed=torch.randn(2, 5),
        meta_embed=torch.randn(2, 3),
    )
    model = ToxicityModel(DummyGNN(), DummyMetaEncoder(), hidden_dim=7)

    output = model(batch)

    assert output.shape == (2, 1)
    assert model.gnn_dim == 5
    assert model.encoder_dim == 3


def test_train_uses_validation_loader_for_early_stopping():
    class ConstantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor([1.0]))

        def forward(self, batch):
            return self.bias.expand(batch.y.shape[0], 1)

    train_loader = [LoaderBatch(y=torch.zeros(4))]
    val_loader = [LoaderBatch(y=torch.zeros(4))]
    model = ConstantModel()

    trained_model, history = train(
        model,
        train_loader,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=None,
        epochs=10,
        device="cpu",
        val_loader=val_loader,
        early_stopping_patience=2,
        verbose_every=0,
    )

    assert trained_model is model
    assert history["monitor_name"] == "val_loss"
    assert history["best_epoch"] == 0
    assert history["stopped_early"] is True
    assert history["epochs_ran"] == 3
    assert len(history["val_loss"]) == history["epochs_ran"]


def test_train_updates_tqdm_progress_bar():
    class ConstantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor([1.0]))

        def forward(self, batch):
            return self.bias.expand(batch.y.shape[0], 1)

    class FakeTqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.kwargs = kwargs
            self.descriptions = []
            self.postfixes = []
            self.messages = []
            self.closed = False

        def __iter__(self):
            return iter(self.iterable)

        def set_description(self, description):
            self.descriptions.append(description)

        def set_postfix(self, postfix):
            self.postfixes.append(dict(postfix))

        def write(self, message):
            self.messages.append(message)

        def close(self):
            self.closed = True

    train_loader = [LoaderBatch(y=torch.zeros(4))]
    val_loader = [LoaderBatch(y=torch.zeros(4))]
    model = ConstantModel()
    progress_bars = []

    def fake_tqdm(iterable, **kwargs):
        progress_bar = FakeTqdm(iterable, **kwargs)
        progress_bars.append(progress_bar)
        return progress_bar

    with patch("src.training.loops.tqdm", fake_tqdm):
        train(
            model,
            train_loader,
            loss_fn=nn.MSELoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            scheduler=None,
            epochs=2,
            device="cpu",
            val_loader=val_loader,
            verbose_every=1,
        )

    assert len(progress_bars) == 1

    progress_bar = progress_bars[0]
    assert progress_bar.kwargs["total"] == 2
    assert progress_bar.kwargs["unit"] == "epoch"
    assert progress_bar.descriptions == ["Epoch 1/2", "Epoch 2/2"]
    assert progress_bar.postfixes[-1]["train_loss"] == "1.0000"
    assert progress_bar.postfixes[-1]["val_loss"] == "1.0000"
    assert progress_bar.postfixes[-1]["val_rmse"] == "1.0000"
    assert progress_bar.postfixes[-1]["val_mae"] == "1.0000"
    assert progress_bar.postfixes[-1]["lr"] == "0.00e+00"
    assert progress_bar.messages == []
    assert progress_bar.closed is True


def test_train_can_log_metrics_to_a_run():
    class ConstantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor([1.0]))

        def forward(self, batch):
            return self.bias.expand(batch.y.shape[0], 1)

    class FakeRun:
        def __init__(self):
            self.logged = []
            self.summary = {}

        def log(self, metrics):
            self.logged.append(dict(metrics))

    train_loader = [LoaderBatch(y=torch.zeros(4))]
    val_loader = [LoaderBatch(y=torch.zeros(4))]
    test_loader = [LoaderBatch(y=torch.zeros(4))]
    model = ConstantModel()
    run = FakeRun()

    _, history = train(
        model,
        train_loader,
        test_loader=test_loader,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=None,
        epochs=2,
        device="cpu",
        val_loader=val_loader,
        verbose_every=0,
        run=run,
    )

    assert len(run.logged) == history["epochs_ran"]
    assert run.logged[0]["epoch"] == 1
    assert run.logged[0]["train/loss"] == 1.0
    assert run.logged[0]["val/loss"] == 1.0
    assert run.logged[0]["val/rmse"] == 1.0
    assert run.logged[0]["val/mae"] == 1.0
    assert run.logged[0]["test/loss"] == 1.0
    assert run.logged[0]["test/rmse"] == 1.0
    assert run.logged[0]["test/mae"] == 1.0
    assert run.logged[0]["optimizer/lr"] == 0.0
    assert run.summary["best_epoch"] == 0
    assert run.summary["monitor_name"] == "val_loss"
    assert run.summary["epochs_ran"] == history["epochs_ran"]
