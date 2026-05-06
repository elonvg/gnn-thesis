import torch
import torch.nn as nn


def _infer_output_dim(module, explicit_dim, candidate_attrs, label):
    if explicit_dim is not None:
        return explicit_dim
    if module is None:
        return 0

    for attr in candidate_attrs:
        value = getattr(module, attr, None)
        if isinstance(value, int):
            return value

    raise ValueError(
        f"Could not infer {label} automatically. "
        f"Pass {label} explicitly or expose one of {candidate_attrs} on the module."
    )


class ToxicityModel(nn.Module):
    def __init__(self, gnn, meta_encoder=None, gnn_dim=None, encoder_dim=None, hidden_dim=64):
        super().__init__()

        self.gnn = gnn
        self.meta_encoder = meta_encoder
        self.gnn_dim = _infer_output_dim(
            gnn,
            gnn_dim,
            candidate_attrs=("out_dim", "out_channels", "output_dim"),
            label="gnn_dim",
        )
        self.encoder_dim = _infer_output_dim(
            meta_encoder,
            encoder_dim,
            candidate_attrs=("output_dim", "out_dim", "out_channels"),
            label="encoder_dim",
        )

        input_dim = self.gnn_dim + self.encoder_dim

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        embeddings = []
        if self.gnn is not None:
            embeddings.append(self.gnn(data))
        if self.meta_encoder is not None:
            embeddings.append(self.meta_encoder(data))

        if len(embeddings) == 1:
            combined = embeddings[0]
        else:
            combined = torch.cat(embeddings, dim=1)

        return self.predictor(combined)
