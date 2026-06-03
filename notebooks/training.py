#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Argparse
import argparse
import random
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--seed", type=int, default=11)
args = parser.parse_args()

fold_id = args.fold
seed = args.seed
print(f"Using fold_id: {fold_id}")
print(f"Using seed: {seed}")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# # Hyperparameters

USE_WANDB = True
n_folds = 5

N_SAMPLES = 1000
# N_SAMPLES = None

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
from src.training.loops import train


pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)

print("Imports imported")


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

graph_featurizer = GraphFeaturizer(ATOM_FEATURES, BOND_FEATURES)

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

NUMERICAL_COLS = MOLECULE_NUMERICAL_COLS


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

folds = 5

smiles_to_cluster = load_butina_clusters(cluster_csv_path, cluster_col)

groups = pd.Series(
    [butina_group_key(graph.smiles, smiles_to_cluster) for graph in graphs],
    name="butina_group"
)

print(f"Rows: {len(groups):,}")
print(f"Unique groups: {groups.nunique():,}")
print(f"Missing group values: {groups.isna().sum():,}")
print(f"Fallback missing-cluster groups: {groups.str.startswith('__missing__::').sum():,}")

group_kfold = GroupKFold(n_splits=folds)

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
from src.models.fragVirtualComboPNAInit import FragVirtualComboPNAInit
from src.models.toxicity_model import ToxicityModel
from src.models.meta_encoder import MetaEncoder, TaxonomyOneHot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########## MODEL HYPERPARAMETERS ##########

PRETRAINED_TAX_DIM = 768 # 768 is the length of the vectors in pretrained_tax_emb.pkl.zip
PRETRAINED_TAXID_OUTPUT_DIM = 256
CATEGORICAL_DIM = 128
NUMERIC_DIM = 128
META_DROPOUT = 0.3

GNN_HIDDEN_DIM = 256
GNN_OUT_DIM = 256
TOWERS = 4

NUM_LAYERS = 3
NUM_TIMESTEPS = 2
DROPOUT = 0.3

FINAL_HIDDEN_DIM = 512

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

    model_gnn = FragVirtualComboPNAInit(
        in_channels=ATOM_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        virtual_edge_dim=VIRTUAL_EDGE_FEATURE_DIM,
        hidden_dim=GNN_HIDDEN_DIM,
        towers=TOWERS,
        deg=pna_deg,
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

epochs = 100
learning_rate = 3e-4
weight_decay = 1e-4
loss_beta = 0.5
early_stopping_patience = 30
early_stopping_min_delta = 1e-4
record_categories = ["species_group", "endpoint", "effect", "conc_unit"]
BATCH_SIZE = globals().get("BATCH_SIZE", 256)
attribute = globals().get("attribute", "species_group")

wandb_run = None

print(f"at fold {fold_id}")

train_idx, val_idx = splits[fold_id]

train_dataset = _build_dataset(graphs, train_idx)
val_dataset = _build_dataset(graphs, val_idx)
test_dataset = None

loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=10,
    factor=0.5,
    min_lr=1e-6,
)

if USE_WANDB:
    wandb_run = wandb.init(
        project="gnn-thesis",
        entity="elonvg-chalmers-university-of-technology",
        job_type="train",
        group=f"{gnn_name}-groupkfold-seed:{seed}",   # same for all folds
        name=f"{gnn_name}-fold-{fold_id}",               # unique per fold
        tags=["notebook", gnn_name],
        config={
            "random_state": seed,
            "n_samples": N_SAMPLES,
            "fold": fold_id,
            "folds": folds,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),

            "filters": FILTERS,
            "split_salt": SPLIT_SALTS,
            "remove_lone": REMOVE_LONE,
            "remove_metals": REMOVE_METALS,
            "max_conc_value": MAX_CONC_VALUE,
            "duration_fill_value": DURATION_FILL_VALUE,
            "max_duration_hours": MAX_DURATION_HOURS,
            
            "num_atom_features": ATOM_FEATURE_DIM,
            "num_bond_features": EDGE_FEATURE_DIM,

            "use_pretrained_taxid": USE_PRETRAINED_TAXID,
            "categorical_cols": CATEGORICAL_COLS,
            "numerical_cols": NUMERICAL_COLS,

            # "split_method": split_method,
            "butina_cluster_col": cluster_col,
            # "stratify_by": stratify_by,
            # "frac_train": frac_train,
            # "frac_valid": frac_valid,
            # "frac_test": frac_test,
            # "target_mean": float(target_mean),
            # "target_std": float(target_std),

            "batch_size": BATCH_SIZE,
            "taxonomy_encoder": TaxonomyOneHot.__name__,
            "gnn_model": f"{gnn_name}-goodtest",
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
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss": loss_fn.__class__.__name__,
            "loss_beta": loss_beta,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        },
    )
    wandb_run.define_metric("epoch")
    metric_prefixes = ["train/*", "val/*", "optimizer/*"]
    if test_loader is not None:
        metric_prefixes.append("test/*")
    for metric_prefix in metric_prefixes:
        wandb_run.define_metric(metric_prefix, step_metric="epoch")
else:
    print("wandb not installed; running without experiment tracking.")

print("Training configuration")
print(f"epochs = {epochs}")
print(f"learning_rate = {learning_rate}")
print(f"weight_decay = {weight_decay}")
print(f"loss = {loss_fn.__class__.__name__}")
print(f"early_stopping_patience = {early_stopping_patience}")

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
    record_categories=CATEGORICAL_COLS,
    record_joint_categories=("endpoint", "species_group"),
    label_encoder=categorical_encoder,
    run=wandb_run,
)

model = model_trained

if wandb_run is not None:
    wandb_run.finish()
    wandb_run = None