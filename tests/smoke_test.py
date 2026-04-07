import torch
from types import SimpleNamespace

from src.models.meta_encoder import (
    CategoricalOneHot,
    MetaEncoder,
    NumericalEncoder,
    TaxonomyOneHot,
)


def build_batch():
    return SimpleNamespace(
        duration=torch.tensor([24.0, 48.0]),
        temperature=torch.tensor([18.0, 21.5]),
        species_group=torch.tensor([1, 2], dtype=torch.long),
        family=torch.tensor([3, 1], dtype=torch.long),
        endpoint=torch.tensor([1, 2], dtype=torch.long),
        effect=torch.tensor([4, 3], dtype=torch.long),
    )


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
