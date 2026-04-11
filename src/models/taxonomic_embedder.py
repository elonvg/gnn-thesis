from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def normalize_taxid(taxid):
    """Normalize raw NCBI taxid values to string keys like '10090'."""
    if taxid is None:
        return None

    if isinstance(taxid, torch.Tensor):
        if taxid.numel() == 0:
            return None
        taxid = taxid.item()

    if pd.isna(taxid):
        return None

    if isinstance(taxid, str):
        normalized = taxid.strip()
        if not normalized or normalized.lower() in {"nan", "none"}:
            return None
        normalized = normalized.split("_")[0]
        if normalized.endswith(".0"):
            normalized = normalized[:-2]
        return normalized or None

    if isinstance(taxid, (int, np.integer)):
        return None if int(taxid) <= 0 else str(int(taxid))

    if isinstance(taxid, (float, np.floating)):
        if np.isnan(taxid) or float(taxid) <= 0:
            return None
        return str(int(taxid))

    normalized = str(taxid).strip()
    return normalized.split("_")[0] or None


def _parse_embedding_string(embedding_text):
    clean_text = str(embedding_text).strip().strip("[]").replace("\n", " ")
    if not clean_text:
        return np.array([], dtype=np.float32)

    vector = np.fromstring(clean_text, sep=" ", dtype=np.float32)
    if vector.size == 0 and "," in clean_text:
        vector = np.fromstring(clean_text, sep=",", dtype=np.float32)
    return vector


@lru_cache(maxsize=None)
def _load_taxonomic_embedding_dict_cached(filepath):
    if filepath.endswith(".pkl") or filepath.endswith(".pkl.zip"):
        series = pd.read_pickle(filepath)
        if hasattr(series, "to_dict"):
            raw_dict = series.to_dict()
        else:
            raw_dict = dict(series)
    else:
        dataframe = pd.read_csv(filepath, compression="infer")

        if {"NCBI_taxa", "taxonomic_embedding"}.issubset(dataframe.columns):
            raw_dict = {
                raw_taxid: _parse_embedding_string(embedding_text)
                for raw_taxid, embedding_text in zip(
                    dataframe["NCBI_taxa"],
                    dataframe["taxonomic_embedding"],
                )
            }
        else:
            with open(filepath, "r", encoding="utf-8") as handle:
                raw_dict = {}
                for line in handle:
                    parts = line.strip().split("\t")
                    if len(parts) != 2:
                        continue
                    raw_taxid, embedding_text = parts
                    raw_dict[raw_taxid] = _parse_embedding_string(embedding_text)

    normalized_dict = {}
    for raw_taxid, embedding in raw_dict.items():
        normalized_taxid = normalize_taxid(raw_taxid)
        if normalized_taxid is None:
            continue

        if isinstance(embedding, torch.Tensor):
            normalized_embedding = embedding.detach().cpu().numpy().astype(np.float32)
        else:
            normalized_embedding = np.asarray(embedding, dtype=np.float32)

        normalized_dict[normalized_taxid] = normalized_embedding

    return normalized_dict


def load_taxonomic_embedding_dict(filepath, return_as_tensor=False):
    """Load pretrained NCBI taxid embeddings from pickle or CSV exports."""
    taxonomic_embedding_dict = dict(_load_taxonomic_embedding_dict_cached(str(filepath)))

    if return_as_tensor:
        return {
            taxid: torch.tensor(embedding, dtype=torch.float32)
            for taxid, embedding in taxonomic_embedding_dict.items()
        }

    return taxonomic_embedding_dict


class TaxonomicEmbedder(nn.Module):
    """
    Look up external taxonomic embeddings and adapt them with a learnable projection.

    Input taxids can be a flat batch like ['9606', '10090'] or a nested batch of lists.
    Missing taxids fall back to a zero vector.
    """

    def __init__(
        self,
        taxonomic_embedding_dict,
        embedding_dim=768,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.taxonomic_embedding_dict = taxonomic_embedding_dict
        self.external_proj = nn.Linear(embedding_dim, embedding_dim)
        self.external_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("zero_embedding", torch.zeros(embedding_dim, dtype=torch.float32))

        nn.init.xavier_uniform_(self.external_proj.weight)
        nn.init.zeros_(self.external_proj.bias)

    def _lookup_single(self, taxid):
        normalized_taxid = normalize_taxid(taxid)
        if normalized_taxid is None:
            return self.zero_embedding.clone()

        embedding = self.taxonomic_embedding_dict.get(normalized_taxid)
        if embedding is None:
            return self.zero_embedding.clone()

        if isinstance(embedding, torch.Tensor):
            return embedding.detach().clone().float()

        return torch.tensor(embedding, dtype=torch.float32)

    def lookup_embeddings(self, taxids):
        if not taxids:
            return torch.empty((0, 1, self.embedding_dim), dtype=torch.float32)

        if taxids[0] is None or isinstance(taxids[0], (str, int, float, np.integer, np.floating, torch.Tensor)):
            embeddings = torch.stack([self._lookup_single(taxid) for taxid in taxids], dim=0)
            return embeddings.unsqueeze(1)

        sample_embeddings = []
        for taxid_list in taxids:
            sample_embeddings.append(torch.stack([self._lookup_single(taxid) for taxid in taxid_list], dim=0))
        return torch.stack(sample_embeddings, dim=0)

    def forward(self, taxids):
        embeddings = self.lookup_embeddings(taxids)
        embeddings = embeddings.to(self.external_proj.weight.device)
        embeddings = self.external_proj(embeddings)
        embeddings = self.external_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PretrainedTaxidEncoder(nn.Module):
    """Batch encoder that reads raw taxids from graph data and returns a fixed-width embedding."""

    def __init__(
        self,
        embedding_path=None,
        taxonomic_embedding_dict=None,
        embedding_dim=768,
        output_dim=64,
        dropout=0.1,
        taxid_field="taxid_raw",
    ):
        super().__init__()

        if taxonomic_embedding_dict is None:
            if embedding_path is None:
                raise ValueError(
                    "PretrainedTaxidEncoder requires either embedding_path or taxonomic_embedding_dict."
                )
            taxonomic_embedding_dict = load_taxonomic_embedding_dict(embedding_path)

        self.taxid_field = taxid_field
        self.output_dim = output_dim
        self.embedder = TaxonomicEmbedder(
            taxonomic_embedding_dict=taxonomic_embedding_dict,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        if output_dim == embedding_dim:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Sequential(
                nn.Linear(embedding_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

    def forward(self, data):
        if not hasattr(data, self.taxid_field):
            raise AttributeError(
                f"Input batch is missing {self.taxid_field!r}, which is required for pretrained taxid embeddings."
            )

        raw_taxids = getattr(data, self.taxid_field)
        if isinstance(raw_taxids, torch.Tensor):
            if raw_taxids.dim() == 0:
                raw_taxids = raw_taxids.unsqueeze(0)
            taxids = raw_taxids.detach().cpu().tolist()
        else:
            taxids = list(raw_taxids)

        embeddings = self.embedder(taxids).squeeze(1)
        return self.output_projection(embeddings)
