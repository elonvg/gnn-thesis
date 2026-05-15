#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Argparse
import argparse

from src.models.gin import GIN

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
args = parser.parse_args()

fold_id = args.fold
print(f"Using fold_id: {fold_id}")


# # Hyperparameters

# In[1]:


USE_WANDB = True
n_folds = 5

N_SAMPLES = 1000  
# N_SAMPLES = None
random_state = 11

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

# In[2]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    import wandb
except ImportError:
    wandb = None

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader
from torch_geometric.utils.smiles import from_smiles

from src.data.io import load_data
from src.data.cleaning import process_data, print_mol_types
from src.data.graph_building import build_graph_features
from src.data.metadata import sequential_encoder, build_config
from src.data.cleaning import fragment_count, is_salt, has_metal, is_single_node
from src.data.splitting import butina_split, show_split_info
from src.data.sampling import LoadData, show_loader_info, display_sampling_effect
from src.training.loops import train
from src.visualization.training_plots import  plot_training, plot_training_metrics, plot_group_training


pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)

print("Imports imported")


# # Load, filter and preprocess data
# 

# In[3]:


DATA_PATH = PROJECT_ROOT / "Data" / "toxicity_all.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

print(f"Data path: {DATA_PATH}")

df_all = load_data(DATA_PATH)

df_processed = process_data(
    df_all,
    n_samples=N_SAMPLES,
    random_state=random_state,
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

print()
print_mol_types(df_processed)

df_processed


# # Encoding Data

# ## Featiruze molecules

# In[4]:


from src.data.featurization import simple_featurizer

# df_processed["features"] = df_processed["SMILES"].apply(from_smiles)

# atom_features = ["atomic_num", "mass"]
# bond_features = ["bond_order"]
df_processed["features"] = df_processed["SMILES"].apply(simple_featurizer)

sample_id = min(16, len(df_processed) - 1)

print()
print(f"{len(df_processed):,} rows with graph features created")


# ## Embeddings

# In[5]:


USE_PRETRAINED_TAXID = True
PRETRAINED_TAXID_PATH = PROJECT_ROOT / "Data" / "moredata" / "pretrained_tax_emb.pkl.zip"
print(f"Pretrained taxid path: {PRETRAINED_TAXID_PATH}")

tax_embedding = {}
df_tax = pd.DataFrame(index=df_processed.index)
config_tax = {}

# Categorical encoding
categorical_cols = [
    "species_group",
    "conc_unit",
    "endpoint", 
    "effect", 
    "is_salt",
    "has_metal",
    "is_single_node",
    ]

df_categorical = df_processed[categorical_cols].copy()
df_categorical, categorical_encoder = sequential_encoder(df_categorical, categorical_cols)
# df_categorical now contains only the sequential data for selected columns

config_categorical = build_config(df_categorical, categorical_cols)

species_group_decoder = {encoded: original for original, encoded in categorical_encoder["species_group"].items()}

print("Categorical embedding config:")
print(config_categorical)
print()

# Numerical encoding 
numerical_cols = [
    "duration",
    "fragment_count",
]

print("Numerical encoding for:")
print(numerical_cols)


# # Build The Final Graph Dataset
# 

# In[6]:


features = build_graph_features(
    df_processed, 
    df_tax, 
    tax_embedding, 
    df_categorical,
    categorical_cols,
    numerical_cols
    )

sample_feature = features[sample_id]

print(f"Graph objects created: {len(features):,}")
print()
print("Info for a sample:")
print(features[sample_id])


# # Prepare for training

# ## Split data

# In[7]:


from sklearn.model_selection import GroupKFold
from src.data.splitting import load_butina_clusters, _build_dataset

cluster_csv_path = PROJECT_ROOT / "Data" / "moredata" / "original" / "butina_cluster_lookup.csv"
cluster_col = "Cluster_at_cutoff_0.2"

folds = 5

smiles_to_cluster = load_butina_clusters(cluster_csv_path, cluster_col)

def butina_group_key(smiles):
    cluster_id = smiles_to_cluster.get(smiles, np.nan)

    if pd.isna(cluster_id):
        return f"__missing__::{smiles}"
    
    if isinstance(cluster_id, float) and cluster_id.is_integer():
        cluster_id = int(cluster_id)

    return f"cluster::{cluster_id}"

groups = pd.Series(
    [butina_group_key(graph.smiles) for graph in features],
    name="butina_group"
)

print(f"Rows: {len(groups):,}")
print(f"Unique groups: {groups.nunique():,}")
print(f"Missing group values: {groups.isna().sum():,}")
print(f"Fallback missing-cluster groups: {groups.str.startswith('__missing__::').sum():,}")

group_kfold = GroupKFold(n_splits=folds)

splits = list(group_kfold.split(features, groups=groups))

train_idx, val_idx = splits[0]

train_dataset = _build_dataset(features, train_idx)
val_dataset = _build_dataset(features, val_idx)
test_dataset = None

show_split_info(train_dataset, val_dataset)


# ## Build DataLoaders
# 

# In[8]:


attribute = "species_group"

train_loader = LoadData(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    attribute=attribute
)

val_loader = LoadData(
    dataset=val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    attribute=attribute,
    target_dataset=train_dataset
)

test_loader = None
if test_dataset is not None:
    test_loader = LoadData(
        dataset=test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        attribute=attribute,
        target_dataset=train_dataset
    )

show_loader_info(attribute, train_loader, val_loader, test_loader, species_group_decoder)

display_sampling_effect(train_dataset, train_loader, "species_group", species_group_decoder)


# # Model and training
# 

# ## Build model

# In[9]:


from src.models.gcn import GCN
from src.models.afp_flex import AFPFlex
from src.models.toxicity_model import ToxicityModel
from src.models.meta_encoder import MetaEncoder, TaxonomyEncoder, TaxonomyOneHot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########## MODEL HYPERPARAMETERS ##########

PRETRAINED_TAX_DIM = 768 # 768 is the length of the vectors in pretrained_tax_emb.pkl.zip
PRETRAINED_TAXID_OUTPUT_DIM = 128
CATEGORICAL_DIM = 16
NUMERIC_DIM = 16
META_DROPOUT = 0.3

GNN_HIDDEN_DIM = 512
GNN_OUT_DIM = 512

NUM_LAYERS = 3
NUM_TIMESTEPS = 2
DROPOUT = 0.3

FINAL_HIDDEN_DIM = 64

ATOM_FEATURE_DIM = features[0].x.shape[1]
EDGE_FEATURE_DIM = features[0].edge_attr.shape[1]

def build_model():
    meta_encoder = MetaEncoder(
        pretrained_taxid_path=PRETRAINED_TAXID_PATH if USE_PRETRAINED_TAXID else None,
        pretrained_tax_dim=PRETRAINED_TAX_DIM,
        pretrained_taxid_output_dim=PRETRAINED_TAXID_OUTPUT_DIM,
        config_categorical=config_categorical,
        categorical_output_dim=CATEGORICAL_DIM,
        numerical_columns=numerical_cols,
        numeric_output_dim=NUMERIC_DIM,
        dropout=META_DROPOUT
    ).to(device)

    # meta_encoder = None

    # model_gnn = AFPFlex(
    #     in_channels=ATOM_FEATURE_DIM,
    #     edge_dim=EDGE_FEATURE_DIM,
    #     hidden_channels=GNN_HIDDEN_DIM,
    #     out_channels=GNN_OUT_DIM,
    #     num_layers=NUM_LAYERS,
    #     num_timesteps=NUM_TIMESTEPS,
    #     dropout=DROPOUT,
    # ).to(device)

    model_gnn = GIN(            
            mol_dim=ATOM_FEATURE_DIM,
            edge_dim=EDGE_FEATURE_DIM,
            num_layers=3,
            hidden_dim=GNN_HIDDEN_DIM,
            output_dim=GNN_OUT_DIM)

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


# ## Train The Model
# 

# In[77]:


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

train_dataset = _build_dataset(features, train_idx)
val_dataset = _build_dataset(features, val_idx)
test_dataset = None

train_loader = LoadData(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    attribute=attribute,
)

val_loader = LoadData(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    attribute=attribute,
    target_dataset=train_dataset,
)

test_loader = None

model, meta_encoder, model_gnn, n_params_meta, n_params_gnn, n_params_total, gnn_name = build_model()
loss_fn = torch.nn.SmoothL1Loss(beta=loss_beta)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        group=f"{gnn_name}-groupkfold-{random_state}",   # same for all folds
        name=f"{gnn_name}-fold-{fold_id}",               # unique per fold
        tags=["notebook", gnn_name],
        config={
            "random_state": random_state,
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

            "tax_embedding": tax_embedding,
            "use_pretrained_taxid": USE_PRETRAINED_TAXID,
            "categorical_cols": categorical_cols,
            "numerical_cols": numerical_cols,

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
            "gnn_model": gnn_name,
            "pretrained_tax_dim": PRETRAINED_TAX_DIM,
            "pretrained_taxid_output_dim": PRETRAINED_TAXID_OUTPUT_DIM,
            "categorical_dim": CATEGORICAL_DIM,
            "numeric_dim": NUMERIC_DIM,
            "meta_dropout": META_DROPOUT,
            "gnn_hidden_dim": GNN_HIDDEN_DIM,
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
    record_categories=categorical_cols,
    record_joint_categories=("endpoint", "species_group"),
    label_encoder=categorical_encoder,
    run=wandb_run,
)

model = model_trained

if wandb_run is not None:
    wandb_run.finish()
    wandb_run = None

# plot_training(history["history_all"])

# plot_training_metrics(history["history_all"])


# ### Wandb finish

# In[15]:


if wandb_run is not None:
    wandb_run.finish()

