import torch
import torch.nn as nn
import torch.nn.functional as F

from .taxonomic_embedder import PretrainedTaxidEncoder


def _num_classes_from_config(value):
    if isinstance(value, tuple):
        return value[0]
    return value


class NumericalEncoder(nn.Module):
    # Numerical encoder for metadata such as duration, fragment_count, is_single_node

    def __init__(self, numerical_columns=None, output_dim=16, dropout=0.2):
        super().__init__()

        self.numerical_columns = list(numerical_columns or ["duration"])
        self.input_dim = len(self.numerical_columns)
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        numerical_values = []

        for col in self.numerical_columns:
            values = getattr(data, col).float()
            if values.dim() == 1:
                values = values.unsqueeze(-1)
            numerical_values.append(values)

        concatenated = torch.cat(numerical_values, dim=-1)
        return self.projection(concatenated)


class TaxonomyEncoder(nn.Module):
    # Embedding for taxonomy data

    def __init__(self, config, output_dim=64, dropout=0.2):
        # config is a dict mapping feature_tax -> (num_unique_values, embedding_dim)
        super().__init__()

        # Create an embedding layer for each taxonomic feature
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_ids, dim) 
            for col, (num_ids, dim) in config.items()
        })
        
        # Calculate the total size of the concatenated vector
        self.raw_dim = sum(dim for _, dim in config.values())

        # Project to fixed output size regardless of which features are selected
        self.projection = nn.Sequential(
            nn.Linear(self.raw_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, data):
        
        embedded_list = []
        for col, emb_layer in self.embeddings.items():
            # Get the IDs for the taxonomic ranks
            ids = getattr(data, col)  # Assuming data has attributes like data.taxid, data.genus, etc.
            embedded_list.append(emb_layer(ids))
        
        # Concatenate all embeddings into one vector
        concatenated = torch.cat(embedded_list, dim=-1)

        # Project to the desired output dimension
        return self.projection(concatenated)


class TaxonomyOneHot(nn.Module):
    # One-hot encoder for taxonomy data

    def __init__(self, config, output_dim=64, dropout=0.2):
        super().__init__()

        self.num_classes = {
            col: _num_classes_from_config(value)
            for col, value in config.items()
        }
        self.raw_dim = sum(self.num_classes.values())

        self.projection = nn.Sequential(
            nn.Linear(self.raw_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        encoded_list = []

        for col, num_classes in self.num_classes.items():
            ids = getattr(data, col).long()
            encoded_list.append(F.one_hot(ids, num_classes=num_classes).float())

        concatenated = torch.cat(encoded_list, dim=-1)
        return self.projection(concatenated)


class CategoricalOneHot(nn.Module):
    # One-hot encoder for metadata such as endpoint, effect, conc_unit etc.

    def __init__(self, config, output_dim=32, dropout=0.2):
        super().__init__()

        self.num_classes = {
            col: _num_classes_from_config(value)
            for col, value in config.items()
        }
        self.raw_dim = sum(self.num_classes.values())

        self.projection = nn.Sequential(
            nn.Linear(self.raw_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        encoded_list = []

        for col, num_classes in self.num_classes.items():
            ids = getattr(data, col).long()
            encoded_list.append(F.one_hot(ids, num_classes=num_classes).float())

        concatenated = torch.cat(encoded_list, dim=-1)
        return self.projection(concatenated)

class MetaEncoder(nn.Module):
    # Main meta encoder

    def __init__(
        self,
        taxonomy_encoder_cls=TaxonomyEncoder,
        config_tax=None,
        tax_output_dim=64,
        pretrained_taxid_encoder_cls=PretrainedTaxidEncoder,
        pretrained_taxid_path=None,
        pretrained_taxid_field="taxid_raw",
        pretrained_taxid_encoder_kwargs=None,
        pretrained_taxid_output_dim=64,
        categorical_encoder_cls=CategoricalOneHot,
        config_categorical=None,
        categorical_output_dim=16,
        numerical_encoder_cls=NumericalEncoder,
        numerical_columns=None,
        numeric_output_dim=16,
        dropout=0.2
    ):
        super().__init__()

        self.tax_encoder = taxonomy_encoder_cls(
            config_tax,
            output_dim=tax_output_dim,
            dropout=dropout
        ) if config_tax else None

        self.categorical_encoder = categorical_encoder_cls(
            config_categorical,
            output_dim=categorical_output_dim,
            dropout=dropout
        ) if config_categorical else None

        self.numeric_encoder = numerical_encoder_cls(
            numerical_columns=numerical_columns,
            output_dim=numeric_output_dim,
            dropout=dropout
        ) if numerical_columns else None

        pretrained_taxid_encoder_kwargs = dict(pretrained_taxid_encoder_kwargs or {})
        if pretrained_taxid_path is not None:
            pretrained_taxid_encoder_kwargs.setdefault("embedding_path", pretrained_taxid_path)
        pretrained_taxid_encoder_kwargs.setdefault("output_dim", pretrained_taxid_output_dim)
        pretrained_taxid_encoder_kwargs.setdefault("taxid_field", pretrained_taxid_field)

        use_pretrained_taxid = (
            pretrained_taxid_encoder_cls is not None
            and (
                pretrained_taxid_path is not None
                or "taxonomic_embedding_dict" in pretrained_taxid_encoder_kwargs
            )
        )
        self.pretrained_taxid_encoder = (
            pretrained_taxid_encoder_cls(**pretrained_taxid_encoder_kwargs)
            if use_pretrained_taxid
            else None
        )

        self.output_dim = 0
        if self.numeric_encoder is not None:
            self.output_dim = self.numeric_encoder.output_dim
        if self.tax_encoder is not None:
            self.output_dim += tax_output_dim
        if self.categorical_encoder is not None:
            self.output_dim += categorical_output_dim
        if self.pretrained_taxid_encoder is not None:
            self.output_dim += self.pretrained_taxid_encoder.output_dim

    def forward(self, data):

        encoded_parts = []

        if self.tax_encoder is not None:
            encoded_parts.append(self.tax_encoder(data))

        if self.pretrained_taxid_encoder is not None:
            encoded_parts.append(self.pretrained_taxid_encoder(data))

        if self.categorical_encoder is not None:
            encoded_parts.append(self.categorical_encoder(data))
        
        if self.numeric_encoder is not None:
            encoded_parts.append(self.numeric_encoder(data))


        return torch.cat(encoded_parts, dim=-1)
