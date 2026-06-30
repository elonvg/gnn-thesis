#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Argparse
import argparse
import hashlib
import json
import random
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--experiment-name", type=str, default=None)
args = parser.parse_args()

fold_id = args.fold
seed = args.seed
experiment_name = args.experiment_name
print(f"Using fold_id: {fold_id}")
print(f"Using seed: {seed}")
print(f"Using experiment_name: {experiment_name}")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# # Hyperparameters

USE_WANDB = True
n_folds = 10

# N_SAMPLES = 100
N_SAMPLES = None

BATCH_SIZE = 1024

FILTERS = {
    "duration_unit": ["h"],
    "effect": ["MOR", "POP", "GRO", "BEH", "REP", "ITX", "PHY", "DVP", "MPH"],
}

SPLIT_SALTS = False
REMOVE_LONE = False
REMOVE_METALS = False

MAX_CONC_VALUE = 10000
DURATION_FILL_VALUE = 1e-6
MAX_DURATION_HOURS = 9000.0


# # Setup

from pathlib import Path
import sys

PROJECT_ROOT = None
for candidate in [Path.cwd(), *Path.cwd().parents]:
    if (candidate / "src").exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    possible = Path.cwd() / "vollmers/gnn-thesis/gnn-thesis"
    if (possible / "src").exists():
        PROJECT_ROOT = possible

if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate the project root.")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

try:
    import wandb
except ImportError:
    wandb = None

from src.data.io import load_data
from src.data.cleaning import process_data
from src.data.metadata import sequential_encoder, build_config
from src.data.sampling import LoadData
from src.training.loops import predict_df, train


pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)

print("Imports imported")


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(_json_ready(key)): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def _relative_to_project(path):
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _config_digest(config, exclude_keys=None):
    exclude_keys = set(exclude_keys or [])
    digest_config = {
        key: value for key, value in config.items()
        if key not in exclude_keys
    }
    config_json = json.dumps(_json_ready(digest_config), sort_keys=True)
    return hashlib.sha1(config_json.encode("utf-8")).hexdigest()[:8]


# # Load, filter and preprocess data

DATA_PATH = PROJECT_ROOT / "Data" / "toxicity_all.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

print(f"Data path: {DATA_PATH}")

df_all = load_data(DATA_PATH)

df_processed = process_data(
    df_all,
    n_samples=N_SAMPLES,
    random_state=seed,
    filters=FILTERS,
    require_duration=False,
    require_taxid=True,
    split_salts=SPLIT_SALTS,
    remove_lone=REMOVE_LONE,
    remove_metals=REMOVE_METALS,
    max_conc_value=MAX_CONC_VALUE,
    duration_fill_value=DURATION_FILL_VALUE,
    max_duration_hours=MAX_DURATION_HOURS,
    log_transform_duration=True,
    keep_duration_raw=True,
)

# Build graphs - node / edge features
from src.data.simple_featurizer import simple_featurizer
from src.data.features_graph import GraphFeaturizer

ATOM_FEATURES = (
    "atomic_num",
    "degree",
    "formal_charge",
    "num_hs",
    "hybridization",
    "is_aromatic",
    "is_in_ring",
    "atomic_mass",
    "period",
    "group",
    "covalent_radius",
    "vdw_radius",
    "is_metal",
)

BOND_FEATURES = (
    "bond_order",
    "is_conjugated",
    "is_in_ring",
    "stereo",
)

graph_featurizer = GraphFeaturizer(ATOM_FEATURES, BOND_FEATURES, add_virtual_edges=False)

df_processed["features"] = df_processed["SMILES"].apply(graph_featurizer.featurize)

graph_cache = graph_featurizer.get_graph_cache()

sample_id = min(16, len(df_processed) - 1)

print()
print(f"{len(df_processed):,} rows with graph features created")
print(f"Total number of unique graphs: {len(graph_cache):,}")

# Molecule features

from src.data.features_mol import add_molecule_metadata

MOLECULE_CATEGORICAL_COLS = [
    "is_salt",
    "has_metal",
    "is_single_node",
]

MOLECULE_NUMERICAL_COLS = [
    "fragment_count",
    # "mol_weight",
    "log10_mol_weight",
    # "logp",
    # "tpsa",
    # "log10_tpsa_plus1",
    # "h_bond_donor_count",
    # "h_bond_acceptor_count",
    # "heavy_atom_count",
    # "log10_heavy_atom_count_plus1",
    # "hetero_atom_count",
    # "halogen_count",
    # "metal_count",
    # "transition_metal_count",
    # "ring_count",
    # "aromatic_ring_count",
    # "rotatable_bond_count",
    "formal_charge",
]

# Add molecule-level metadata for categorical and numerical encoders
df_processed = add_molecule_metadata(df_processed, categorical_cols=MOLECULE_CATEGORICAL_COLS, numerical_cols=MOLECULE_NUMERICAL_COLS)


# Encode features

USE_PRETRAINED_TAXID = True
PRETRAINED_TAXID_PATH = PROJECT_ROOT / "Data" / "moredata" / "pretrained_tax_emb.pkl.zip"
print(f"Pretrained taxid path: {PRETRAINED_TAXID_PATH}")

config_tax = {}

# Categorical encoding
exp_categorical_cols = [
    "species_group",
    "conc_unit",
    "endpoint", 
    "effect"
]

CATEGORICAL_COLS = exp_categorical_cols + MOLECULE_CATEGORICAL_COLS

df_categorical = df_processed[CATEGORICAL_COLS].copy()
df_categorical, categorical_encoder = sequential_encoder(df_categorical, CATEGORICAL_COLS)
# df_categorical now contains only the sequential data for selected columns

config_categorical = build_config(df_categorical, CATEGORICAL_COLS)

species_group_decoder = {encoded: original for original, encoded in categorical_encoder["species_group"].items()}

print("Categorical embedding config:")
print(config_categorical)

NUMERICAL_COLS = ["duration"] + MOLECULE_NUMERICAL_COLS


# Appebd metadata to graphs

from src.data.graph_building import build_graphs
graphs = build_graphs(
    df_processed,  
    df_categorical,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    )

sample_graph = graphs[sample_id]

print(f"Graph objects created: {len(graphs):,}")
print()
print("Info for a sample graph:")
print(graphs[sample_id])


# # Prepare for training

# Split data

from sklearn.model_selection import GroupKFold
from src.data.splitting import load_butina_clusters, _build_dataset, butina_group_key

cluster_csv_path = PROJECT_ROOT / "Data" / "moredata" / "original" / "butina_cluster_lookup.csv"
cluster_col = "Cluster_at_cutoff_0.2"

smiles_to_cluster = load_butina_clusters(cluster_csv_path, cluster_col)

groups = pd.Series(
    [butina_group_key(graph.smiles, smiles_to_cluster) for graph in graphs],
    name="butina_group"
)

print(f"Rows: {len(groups):,}")
print(f"Unique groups: {groups.nunique():,}")
print(f"Missing group values: {groups.isna().sum():,}")
print(f"Fallback missing-cluster groups: {groups.str.startswith('__missing__::').sum():,}")

group_kfold = GroupKFold(n_splits=n_folds)

splits = list(group_kfold.split(graphs, groups=groups))

train_idx, val_idx = splits[fold_id]

train_dataset = _build_dataset(graphs, train_idx)
val_dataset = _build_dataset(graphs, val_idx)
test_dataset = None



# Build DataLoaders
attribute = "species_group"

train_loader = LoadData(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler_type="weighted",
    shuffle=False, 
    attribute=attribute
)

val_loader = LoadData(
    dataset=val_dataset, 
    batch_size=BATCH_SIZE, 
    sampler_type="sequential",
    shuffle=False, 
    attribute=attribute,
    target_dataset=train_dataset
)

test_loader = None
if test_dataset is not None:
    test_loader = LoadData(
        dataset=test_dataset, 
        batch_size=BATCH_SIZE, 
        sampler_type="sequential",
        shuffle=False, 
        attribute=attribute,
        target_dataset=train_dataset
    )

# # Model and training

# Build model

from src.models.pna import PNA, PNA_1_5M_CONFIG, compute_pna_degree_histogram
from src.models.toxicity_model import ToxicityModel
from src.models.meta_encoder import MetaEncoder, TaxonomyOneHot
from src.models.attentive_fp import AttentiveFP
from src.models.afpGAT import AFPGAT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########## MODEL HYPERPARAMETERS ##########

PRETRAINED_TAX_DIM = 768 # 768 is the length of the vectors in pretrained_tax_emb.pkl.zip
PRETRAINED_TAXID_OUTPUT_DIM = 512
CATEGORICAL_DIM = 128
NUMERIC_DIM = 128
META_DROPOUT = 0.3

GNN_HIDDEN_DIM = 512
GNN_OUT_DIM = 512
TOWERS = 4

NUM_LAYERS = 4
NUM_TIMESTEPS = 2
DROPOUT = 0.3

FINAL_HIDDEN_DIM = 1024

ATOM_FEATURE_DIM = graphs[0].x.shape[1]
EDGE_FEATURE_DIM = graphs[0].edge_attr.shape[1]
VIRTUAL_EDGE_FEATURE_DIM = graphs[0].virtual_edge_attr.shape[1] if hasattr(graphs[0], "virtual_edge_attr") else 0

def build_model():
    meta_encoder = MetaEncoder(
        pretrained_taxid_path=PRETRAINED_TAXID_PATH if USE_PRETRAINED_TAXID else None,
        pretrained_tax_dim=PRETRAINED_TAX_DIM,
        pretrained_taxid_output_dim=PRETRAINED_TAXID_OUTPUT_DIM,
        config_categorical=config_categorical,
        categorical_output_dim=CATEGORICAL_DIM,
        numerical_columns=NUMERICAL_COLS,
        numeric_output_dim=NUMERIC_DIM,
        dropout=META_DROPOUT
    ).to(device)

    # meta_encoder = None

    pna_deg = compute_pna_degree_histogram(train_dataset)

    model_gnn = AFPGAT(
        in_channels=ATOM_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        hidden_dim=GNN_HIDDEN_DIM,
        out_dim=GNN_OUT_DIM,
        num_layers=NUM_LAYERS,
        num_timesteps=NUM_TIMESTEPS,
        dropout=DROPOUT,
    ).to(device)

    # model_gnn = None

    model = ToxicityModel(
        model_gnn,
        meta_encoder,
        hidden_dim=FINAL_HIDDEN_DIM,
    ).to(device)

    n_params_meta = sum(p.numel() for p in meta_encoder.parameters() if p.requires_grad) if meta_encoder else 0
    n_params_gnn = sum(p.numel() for p in model_gnn.parameters() if p.requires_grad) if model_gnn else 0
    n_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gnn_name = type(model_gnn).__name__ if model_gnn is not None else "metadata_only"

    return model, meta_encoder, model_gnn, n_params_meta, n_params_gnn, n_params_total, gnn_name

model, meta_encoder, model_gnn, n_params_meta, n_params_gnn, n_params_total, gnn_name = build_model()


print(f"Device: {device}")
print(f"Meta encoder trainable parameters: {n_params_meta:,}")
print(f"GNN trainable parameters: {n_params_gnn:,}")
print(f"Total trainable parameters: {n_params_total:,}")
print()
print(model)


# Train The Model

epochs = 150
learning_rate = 3e-4
weight_decay = 1e-4
early_stopping_patience = 150
early_stopping_min_delta = 1e-4
record_categories = ["species_group", "endpoint", "effect", "conc_unit"]
BATCH_SIZE = globals().get("BATCH_SIZE", 1024)
attribute = globals().get("attribute", "species_group")
mixed_precision = device.type == "cuda"
amp_dtype = "float16"
RESTORE_BEST_MODEL = False
PREDICTION_MODEL_STATE = "best_val" if RESTORE_BEST_MODEL else "final_epoch"

wandb_run = None

print(f"at fold {fold_id}")

loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=10,
    factor=0.5,
    min_lr=1e-6,
)

run_group = f"{gnn_name}-groupkfold-seed:{seed}"
run_name = f"{gnn_name}-fold-{fold_id}"
run_config = {
    "random_state": seed,
    "n_samples": N_SAMPLES,
    "fold": fold_id,
    "folds": n_folds,
    "train_size": len(train_dataset),
    "val_size": len(val_dataset),
    "total_processed_size": len(df_processed),

    "data_path": _relative_to_project(DATA_PATH),
    "filters": FILTERS,
    "split_salt": SPLIT_SALTS,
    "remove_lone": REMOVE_LONE,
    "remove_metals": REMOVE_METALS,
    "max_conc_value": MAX_CONC_VALUE,
    "duration_fill_value": DURATION_FILL_VALUE,
    "max_duration_hours": MAX_DURATION_HOURS,
    "log_transform_duration": True,
    "keep_duration_raw": True,

    "num_atom_features": ATOM_FEATURE_DIM,
    "num_bond_features": EDGE_FEATURE_DIM,
    "num_virtual_edge_features": VIRTUAL_EDGE_FEATURE_DIM,
    "atom_features": ATOM_FEATURES,
    "bond_features": BOND_FEATURES,
    "mol_categorical_cols": MOLECULE_CATEGORICAL_COLS,
    "mol_numerical_cols": MOLECULE_NUMERICAL_COLS,

    "use_pretrained_taxid": USE_PRETRAINED_TAXID,
    "pretrained_taxid_path": _relative_to_project(PRETRAINED_TAXID_PATH),
    "categorical_cols": CATEGORICAL_COLS,
    "numerical_cols": NUMERICAL_COLS,
    "categorical_embedding_config": config_categorical,
    "categorical_encoder": categorical_encoder,

    "split_method": "GroupKFold",
    "split_group": "butina_cluster",
    "butina_cluster_path": _relative_to_project(cluster_csv_path),
    "butina_cluster_col": cluster_col,

    "batch_size": BATCH_SIZE,
    "sampler_attribute": attribute,
    "train_sampler_type": "weighted",
    "val_sampler_type": "sequential",
    "taxonomy_encoder": TaxonomyOneHot.__name__,
    "gnn_model": f"{gnn_name}-16M-10f",
    "pretrained_tax_dim": PRETRAINED_TAX_DIM,
    "pretrained_taxid_output_dim": PRETRAINED_TAXID_OUTPUT_DIM,
    "categorical_dim": CATEGORICAL_DIM,
    "numeric_dim": NUMERIC_DIM,
    "meta_dropout": META_DROPOUT,
    "gnn_hidden_dim": GNN_HIDDEN_DIM,
    "gnn_towers": TOWERS,
    "gnn_out_dim": GNN_OUT_DIM,
    "num_layers": NUM_LAYERS,
    "num_timesteps": NUM_TIMESTEPS,
    "dropout": DROPOUT,
    "final_hidden_dim": FINAL_HIDDEN_DIM,
    "n_params_meta": n_params_meta,
    "n_params_gnn": n_params_gnn,
    "n_params_total": n_params_total,

    "epochs": epochs,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "loss": loss_fn.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "scheduler": scheduler.__class__.__name__,
    "scheduler_mode": "min",
    "scheduler_patience": 10,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,
    "mixed_precision": mixed_precision,
    "amp_dtype": amp_dtype,
    "early_stopping_patience": early_stopping_patience,
    "early_stopping_min_delta": early_stopping_min_delta,
    "eval_every": 2,
    "record_categories": record_categories,
    "record_joint_categories": ("endpoint", "species_group"),
    "restore_best_model": RESTORE_BEST_MODEL,
    "prediction_model_state": PREDICTION_MODEL_STATE,
    "wandb_group": run_group,
    "wandb_name": run_name,
}

config_hash = _config_digest(
    run_config,
    exclude_keys={"fold", "train_size", "val_size", "wandb_name"},
)
experiment_id = experiment_name or f"{gnn_name}_groupkfold_seed{seed}_{config_hash}"
run_config["experiment_id"] = experiment_id

if USE_WANDB and wandb is not None:
    wandb_run = wandb.init(
        project="gnn-thesis",
        entity="elonvg-chalmers-university-of-technology",
        job_type="train",
        group=run_group,
        name=run_name,
        tags=["notebook", gnn_name],
        config=_json_ready(run_config),
    )
    wandb_run.define_metric("epoch")
    metric_prefixes = ["train/*", "val/*", "optimizer/*"]
    if test_loader is not None:
        metric_prefixes.append("test/*")
    for metric_prefix in metric_prefixes:
        wandb_run.define_metric(metric_prefix, step_metric="epoch")
elif USE_WANDB:
    print("wandb not installed; running without experiment tracking.")
else:
    print("W&B disabled; running without experiment tracking.")

print("Training configuration")
print(f"epochs = {epochs}")
print(f"learning_rate = {learning_rate}")
print(f"weight_decay = {weight_decay}")
print(f"loss = {loss_fn.__class__.__name__}")
print(f"mixed_precision = {mixed_precision}")
print(f"amp_dtype = {amp_dtype}")
print(f"early_stopping_patience = {early_stopping_patience}")
print(f"restore_best_model = {RESTORE_BEST_MODEL}")
print(f"prediction_model_state = {PREDICTION_MODEL_STATE}")

model_trained, history = train(
    model,
    train_loader,
    test_loader=test_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=epochs,
    device=device,
    early_stopping_patience=early_stopping_patience,
    early_stopping_min_delta=early_stopping_min_delta,
    record_categories=record_categories,
    record_joint_categories=("endpoint", "species_group"),
    label_encoder=categorical_encoder,
    eval_every=2,
    run=wandb_run,
    mixed_precision=mixed_precision,
    amp_dtype=amp_dtype,
    restore_best_model=RESTORE_BEST_MODEL,
)

model = model_trained

artifact_dir = PROJECT_ROOT / "outputs" / "reports" / "model_experiments" / experiment_id
artifact_dir.mkdir(parents=True, exist_ok=True)

val_predictions = predict_df(
    model,
    val_loader,
    device=device,
    cols=["row_id"],
).rename(
    columns={
        "pred_norm": "pred_log10c",
        "actual_norm": "actual_log10c",
    }
)
val_predictions["row_id"] = val_predictions["row_id"].astype(int)

base_results_df = df_processed.drop(columns=["features"], errors="ignore").copy()
base_results_df.insert(0, "row_id", base_results_df.index.astype(int))

base_results_df["butina_group"] = base_results_df["row_id"].map(groups.to_dict())

val_results_df = (
    base_results_df
    .merge(
        val_predictions[["row_id", "actual_log10c", "pred_log10c"]],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    .sort_values("row_id")
    .reset_index(drop=True)
)
val_results_df.insert(1, "fold", fold_id)
val_results_df.insert(2, "split", "val")
val_results_df.insert(3, "experiment_id", experiment_id)
val_results_df["abs_error_log10c"] = (
    val_results_df["pred_log10c"] - val_results_df["actual_log10c"]
).abs()

expected_val_rows = len(val_dataset)
if len(val_results_df) != expected_val_rows:
    raise RuntimeError(
        f"Expected {expected_val_rows} validation predictions, "
        f"but saved dataframe has {len(val_results_df)} rows."
    )

predictions_path = artifact_dir / f"fold_{fold_id:02d}_val_predictions.csv.gz"
config_path = artifact_dir / f"fold_{fold_id:02d}_config.json"
val_results_df.to_csv(predictions_path, index=False, compression="gzip")

history_all = history.get("history_all", {})
best_epoch_index = history_all.get("best_epoch")
artifact_config = {
    **run_config,
    "artifact_dir": _relative_to_project(artifact_dir),
    "predictions_path": _relative_to_project(predictions_path),
    "config_path": _relative_to_project(config_path),
    "validation_prediction_rows": len(val_results_df),
    "prediction_epoch": history_all.get("epochs_ran"),
    "epochs_ran": history_all.get("epochs_ran"),
    "best_epoch_index": best_epoch_index,
    "best_epoch": None if best_epoch_index is None else best_epoch_index + 1,
    "best_monitor_value": history_all.get("best_monitor_value"),
    "monitor_name": history_all.get("monitor_name"),
    "stopped_early": history_all.get("stopped_early"),
    "restored_best_model": history_all.get("restored_best_model"),
}

with config_path.open("w", encoding="utf-8") as config_file:
    json.dump(_json_ready(artifact_config), config_file, indent=2, sort_keys=True)

print(f"Saved validation predictions: {predictions_path}")
print(f"Saved run config: {config_path}")

if wandb_run is not None:
    wandb_run.summary.update(
        {
            "experiment_id": experiment_id,
            "predictions_path": _relative_to_project(predictions_path),
            "config_path": _relative_to_project(config_path),
            "prediction_model_state": PREDICTION_MODEL_STATE,
            "validation_prediction_rows": len(val_results_df),
        }
    )

checkpoint_dir = PROJECT_ROOT / "outputs" / "models" / "afpGAT-16M"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = checkpoint_dir / f"{gnn_name}-16M-fold{fold_id}.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_epoch": history["history_all"]["best_epoch"],
        "best_monitor_value": history["history_all"]["best_monitor_value"],

        "model_config": {
            "use_pretrained_taxid": USE_PRETRAINED_TAXID,
            "pretrained_taxid_path": str(PRETRAINED_TAXID_PATH),
            "pretrained_tax_dim": PRETRAINED_TAX_DIM,
            "pretrained_taxid_output_dim": PRETRAINED_TAXID_OUTPUT_DIM,
            "categorical_dim": CATEGORICAL_DIM,
            "numeric_dim": NUMERIC_DIM,
            "meta_dropout": META_DROPOUT,
            "gnn_hidden_dim": GNN_HIDDEN_DIM,
            "gnn_out_dim": GNN_OUT_DIM,
            "towers": TOWERS,
            "num_layers": NUM_LAYERS,
            "num_timesteps": NUM_TIMESTEPS,
            "dropout": DROPOUT,
            "final_hidden_dim": FINAL_HIDDEN_DIM,
            "atom_feature_dim": ATOM_FEATURE_DIM,
            "edge_feature_dim": EDGE_FEATURE_DIM,
            "virtual_edge_feature_dim": VIRTUAL_EDGE_FEATURE_DIM,
        },
    },
    checkpoint_path,
)

print(f"Saved model to {checkpoint_path}")

if wandb_run is not None:
    wandb_run.finish()
    wandb_run = None
