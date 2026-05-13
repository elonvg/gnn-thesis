#!/usr/bin/env python
# coding: utf-8

# # Setup


n_folds = 4
MAX_ROWS = 100000  # set to an integer like 15000 for faster experiments
# MAX_ROWS = None

from pathlib import Path
import sys

PROJECT_ROOT = None
for candidate in [Path.cwd(), *Path.cwd().parents]:
    if (candidate / "src").exists():
        PROJECT_ROOT = candidate
        break

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
from src.data.cleaning import print_mol_types
from src.data.graph_building import build_graph_features
from src.data.metadata import sequential_encoder, build_config
from src.data.cleaning import mask_data, preprocess
from src.data.cleaning import fragment_count, is_salt, has_metal, is_single_node
from src.data.splitting import butina_split, show_split_info
from src.data.sampling import LoadData, show_loader_info, display_sampling_effect
from src.training.loops import train
from src.visualization.training_plots import  plot_training, plot_training_metrics, plot_group_training


pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)

DATA_PATH = PROJECT_ROOT / "Data" / "toxicity_all.csv"

print("Setup complete")
print(f"Data file: {DATA_PATH}")


# # Load And Filter Data
# 

# In[2]:


selected_columns = [
    'SK_unique_id',
    'species_common_name',
    'species_latin_name',
    'CAS',
    'chemical_name',
    'conc_unit',
    'conc',
    'duration',
    'duration_unit',
    'effect',
    'endpoint',
    'SMILES',
    'organism_lifestage_categorized',
    'administration_route_categorized',
    'NCBI_sci_name',
    'NCBI_last_known_rank',
    'NCBI_rank_superkingdom',
    'NCBI_rank_kingdom',
    'NCBI_rank_phylum',
    'NCBI_rank_subphylum',
    'NCBI_rank_class',
    'NCBI_rank_order',
    'NCBI_rank_family',
    'NCBI_rank_genus',
    'NCBI_rank_species',
    'species_group_corrected'
]

df_all = load_data(DATA_PATH, selected_columns)

df_all = df_all.rename(columns={
    'species_group_corrected': 'species_group',
    'organism_lifestage_categorized': 'organism_lifestage',
    'administration_route_categorized': 'administration_route'
})

# Rename columns starting with NCBI_ to be more concise
df_all = df_all.rename(columns={
    'NCBI_rank_superkingdom': 'superkingdom',
    'NCBI_rank_kingdom': 'kingdom',
    'NCBI_rank_phylum': 'phylum',
    'NCBI_rank_subphylum': 'subphylum',
    'NCBI_rank_class': 'class',
    'NCBI_rank_order': 'order',
    'NCBI_rank_family': 'family',
    'NCBI_rank_genus': 'genus',
    'NCBI_rank_species': 'species',
    'NCBI_sci_name': 'species_sci_name',
    'NCBI_last_known_rank': 'taxid'
})

# Fill missing organimsm_lifestage -> adult, administration_route -> fill, duration_unit -> h
df_all['organism_lifestage'] = df_all['organism_lifestage'].fillna('adult')
df_all['administration_route'] = df_all['administration_route'].fillna('fill')
df_all['duration_unit'] = df_all['duration_unit'].fillna('h')

# Filters
filters = {
    # "conc_unit": ["mg/L"],
    "duration_unit": ["h"],
    # "endpoint": ["EC50"],
    "effect": ["MOR", "POP", "GRO", "BEH", "REP", "ITX", "PHY", "DVP", "MPH"],
}
require_duration = False
require_taxonomy = True

taxonomy_cols = (
    "class",
    "family",
    "genus",
    "species",
)

# Create mask
mask = mask_data(
    df_all,
    filters=filters,
    require_duration=require_duration,
    require_taxonomy=require_taxonomy,
    taxonomy_columns=taxonomy_cols,
)

# Apply mask and filter
df_filtered = df_all.loc[mask].copy()

# Convert taxonomy columns to numeric, coercing errors to NaN and then to nullable Int64
for col in taxonomy_cols:
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce").astype("Int64")

print()
print("Loaded and filtered training data")
print(f"Rows in full data: {len(df_all):,}")
print(f"Rows after filter: {len(df_filtered):,}")
print()
print("Filtered preview")
print(df_filtered.head())


# # Preprocess Molecules And Targets
# 

# In[5]:


# Cut dataset for faster experiments

random_state = 11

# Optionally limit the number of rows for faster experiments
if MAX_ROWS is not None and len(df_filtered) > MAX_ROWS:
    df_filtered = df_filtered.sample(n=MAX_ROWS, random_state=random_state).reset_index(drop=True)
else:
    df_filtered = df_filtered.reset_index(drop=True)


# Preprocess data
SPLIT_SALTS = False
REMOVE_LONE = False
REMOVE_METALS = False

MAX_CONC_VALUE = 10000
DURATION_FILL_VALUE = 1e-6
MAX_DURATION_HOURS = 9000.0
LOG_TRANSFORM_DURATION = True

df_processed = preprocess(
    df_filtered.copy(),
    split_salts=False,
    remove_lone=False,
    remove_metals=False,
    max_conc_value=MAX_CONC_VALUE,
    duration_fill_value=DURATION_FILL_VALUE,
    max_duration_hours=MAX_DURATION_HOURS,
    log_transform_duration=True,
    keep_duration_raw=True,
)

print(f"Rows before preprocessing: {len(df_filtered):,}")
print(f"Rows after preprocessing:  {len(df_processed):,}")
print(f"Rows removed: {len(df_filtered) - len(df_processed):,}")
print()
print("Preprocessed preview")
print(df_processed[["SMILES", "duration_raw", "duration", "conc", "log10c", "species_group"]].head())
print()
print_mol_types(df_processed)


# # Encoding Data

# ## Featiruze molecules

# In[6]:


from src.data.featurization import simple_featurizer

# df_processed["features"] = df_processed["SMILES"].apply(from_smiles)

# atom_features = ["atomic_num", "mass"]
# bond_features = ["bond_order"]
df_processed["features"] = df_processed["SMILES"].apply(simple_featurizer)

sample_id = min(16, len(df_processed) - 1)

print()
print(f"{len(df_processed):,} rows with graph features created")


# ## Taxonomy Encoding

# In[7]:


USE_PRETRAINED_TAXID = True
PRETRAINED_TAXID_PATH = PROJECT_ROOT / "Data" / "moredata" / "pretrained_tax_emb.pkl.zip"

tax_embedding = {
    "taxid": 16,
    # "species_group": 16,
    # "genus": 8,
    # "family": 8,
    # "class": 4,
}

# Remove taxid from embedding if using pretrained taxid embeddings
model_tax_embedding = (
    {key: value for key, value in tax_embedding.items() if key != "taxid"}
    if USE_PRETRAINED_TAXID
    else tax_embedding
)

df_tax = df_processed[list(tax_embedding.keys())].copy()
df_tax, tax_encoders = sequential_encoder(df_tax, tax_embedding.keys())
# df_tax now contains only the sequential data for selected columns

config_tax = build_config(df_tax, model_tax_embedding)
taxid_decoder = {encoded: original for original, encoded in tax_encoders["taxid"].items()}

# species_decoder = {encoded: original for original, encoded in tax_encoders["species_group"].items()}

if not USE_PRETRAINED_TAXID:
    print("Don't use pretrained taxid embeddings")
    print("Taxonomy embedding config")
    print(config_tax)


if USE_PRETRAINED_TAXID:
    print(f"Use pretrained taxid embeddings")
    print(f"Pretrained taxid path: {PRETRAINED_TAXID_PATH}")


# ## Categorical and Numerical Encoding

# In[8]:


# Add some more metadata cols
df_processed["fragment_count"] = df_processed["SMILES"].apply(fragment_count).astype(float)
df_processed["is_salt"] = df_processed["SMILES"].apply(is_salt).astype(float)
df_processed["has_metal"] = df_processed["SMILES"].apply(has_metal).astype(float)
df_processed["is_single_node"] = df_processed["SMILES"].apply(is_single_node).astype(float)

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
    # "is_salt",
    # "has_metal",
    # "is_single_node",
]

print("Numerical encoding for:")
print(numerical_cols)


# # Build The Final Graph Dataset
# 

# In[9]:


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

# In[10]:


from sklearn.model_selection import GroupKFold
from src.data.splitting import load_butina_clusters, _build_dataset

cluster_csv_path = PROJECT_ROOT / "Data" / "moredata" / "original" / "butina_cluster_lookup.csv"

cluster_col = "Cluster_at_cutoff_0.2"

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

group_kfold = GroupKFold(n_splits=n_folds)

splits = list(group_kfold.split(features, groups=groups))

train_idx, val_idx = splits[0]

train_dataset = _build_dataset(features, train_idx)
val_dataset = _build_dataset(features, val_idx)
test_dataset = None

show_split_info(train_dataset, val_dataset)


# ## Build DataLoaders
# 

# In[11]:


BATCH_SIZE = 256
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

# In[12]:


from src.models.attentive_fp import AttentiveFP
from src.models.afp_flex import AFPFlex
from src.models.toxicity_model import ToxicityModel
from src.models.meta_encoder import MetaEncoder, TaxonomyEncoder, TaxonomyOneHot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAX_DIM = 16
PRETRAINED_TAX_DIM = 768 # 768 is the length of the vectors in pretrained_tax_emb.pkl.zip
PRETRAINED_TAXID_OUTPUT_DIM = 128
CATEGORICAL_DIM = 16
NUMERIC_DIM = 16
META_DROPOUT = 0.3

GNN_HIDDEN_DIM = 64
GNN_OUT_DIM = 64

NUM_LAYERS = 3
NUM_TIMESTEPS = 2
DROPOUT = 0.3

FINAL_HIDDEN_DIM = 64

ATOM_FEATURE_DIM = features[0].x.shape[1]
EDGE_FEATURE_DIM = features[0].edge_attr.shape[1]

def build_model():
    meta_encoder = MetaEncoder(
        taxonomy_encoder_cls=TaxonomyOneHot,
        config_tax=config_tax,
        tax_output_dim=TAX_DIM,
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

    model_gnn = AFPFlex(
        in_channels=ATOM_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        hidden_channels=GNN_HIDDEN_DIM,
        out_channels=GNN_OUT_DIM,
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


# ## Train The Model
# 

# In[13]:


from src.visualization.training_plots import plot_training_metrics

epochs = 100
learning_rate = 3e-4
weight_decay = 1e-4
loss_beta = 0.5
early_stopping_patience = 30
early_stopping_min_delta = 1e-4
record_categories = ["species_group", "endpoint", "effect", "conc_unit"]
BATCH_SIZE = globals().get("BATCH_SIZE", 256)
attribute = globals().get("attribute", "species_group")

USE_WANDB = wandb is not None  # set to False to skip tracking
# USE_WANDB = False  # set to False to skip tracking
wandb_run = None

for fold_id in range(n_folds):

    print("at fold {fold_id}")

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
            name=f"100k-meanpool-fold-{fold_id}",               # unique per fold
            tags=["notebook", gnn_name],
            config={
                "random_state": random_state,
                "n_samples": MAX_ROWS,
                "fold": fold_id,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),

                "filters": filters,
                "split_salt": SPLIT_SALTS,
                "remove_lone": REMOVE_LONE,
                "remove_metals": REMOVE_METALS,
                "max_conc_value": MAX_CONC_VALUE,
                "duration_fill_value": DURATION_FILL_VALUE,
                "max_duration_hours": MAX_DURATION_HOURS,
                "log_transform_duration": LOG_TRANSFORM_DURATION,

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
                "tax_dim": TAX_DIM,
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


# # Results

# ## Check Overall Evaluation Performance

# In[ ]:


from src.training.loops import predict_df
from src.visualization.result_plots import summarize_by_group, plot_group_mae

analysis_df = df_processed.reset_index(drop=True)[[
    "species_group",
    "species_latin_name",
    "endpoint",
    "effect",
    "fragment_count",
    "has_metal",
    "is_single_node",
    "is_salt",
    "conc_unit"
]].copy()

analysis_df["fragment_bin"] = pd.cut(analysis_df["fragment_count"], [-1, 1, 2, np.inf], labels=["1", "2", "3+"])
analysis_df["has_metal_group"] = analysis_df["has_metal"].map({0.0: "No metal", 1.0: "Has metal"})
analysis_df["is_single_group"] = analysis_df["is_single_node"].map({0.0: "Not single-node", 1.0: "Single-node"})
analysis_df["is_salt"] = analysis_df["is_salt"].map({0.0: "Not salt", 1.0: "Is salt"})

eval_loader = test_loader if test_loader is not None else val_loader
eval_name = "test" if test_loader is not None else "val"

results_df = predict_df(model, eval_loader, device, cols=["row_id", "smiles", "taxid_raw"])
results_df["row_id"] = results_df["row_id"].astype(int)
results_df["taxid"] = results_df["taxid_raw"].astype(int)
results_df = results_df.drop(columns="taxid_raw").join(analysis_df, on="row_id")

results_df["pred_log10c"] = results_df["pred_norm"] # * target_std + target_mean
results_df["actual_log10c"] = results_df["actual_norm"] # * target_std + target_mean
results_df["residual_log10c"] = results_df["pred_log10c"] - results_df["actual_log10c"]
results_df["abs_error_log10c"] = results_df["residual_log10c"].abs()
results_df["pred_conc"] = 10 ** results_df["pred_log10c"]
results_df["actual_conc"] = 10 ** results_df["actual_log10c"]
results_df["fold_error"] = np.maximum(
    results_df["pred_conc"] / results_df["actual_conc"],
    results_df["actual_conc"] / results_df["pred_conc"],
)

train_df = analysis_df.iloc[[g.row_id.item() for g in train_dataset]].copy()

summary_metrics = {
    "r2_norm": r2_score(results_df["actual_norm"], results_df["pred_norm"]),
    "r2_log10c": r2_score(results_df["actual_log10c"], results_df["pred_log10c"]),
    "rmse_log10c": mean_squared_error(results_df["actual_log10c"], results_df["pred_log10c"]) ** 0.5,
    "mae_log10c": mean_absolute_error(results_df["actual_log10c"], results_df["pred_log10c"]),
    "median_fold_error": results_df["fold_error"].median(),
}

print(f"Overall {eval_name}-set metrics")
print(f"R^2 (normalized target): {summary_metrics['r2_norm']:.3f}")
print(f"R^2 (log10c): {summary_metrics['r2_log10c']:.3f}")
print(f"RMSE (log10c): {summary_metrics['rmse_log10c']:.3f}")
print(f"MAE (log10c): {summary_metrics['mae_log10c']:.3f}")
print(f"Median fold error (conc scale): {summary_metrics['median_fold_error']:.3f}")

largest_errors = results_df[[
    "species_latin_name",
    "species_group",
    "endpoint",
    "effect",
    "actual_log10c",
    "pred_log10c",
    "abs_error_log10c",
    "fold_error",
    "smiles",
]].sort_values("abs_error_log10c", ascending=False).head(10)

largest_errors


# ## Visual Result Checks

# In[14]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(results_df["actual_log10c"], results_df["pred_log10c"], alpha=0.25)
plot_min = min(results_df["actual_log10c"].min(), results_df["pred_log10c"].min())
plot_max = max(results_df["actual_log10c"].max(), results_df["pred_log10c"].max())
axes[0].plot([plot_min, plot_max], [plot_min, plot_max], "r--")
axes[0].set_xlabel("Actual log10c")
axes[0].set_ylabel("Predicted log10c")
axes[0].set_title("Prediction vs actual")

axes[1].hist(results_df["residual_log10c"], bins=40, alpha=0.85)
axes[1].axvline(0, color="r", linestyle="--")
axes[1].set_xlabel("Residual (pred - actual)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual distribution")

plt.tight_layout()
plt.show()


# ## Performance By Group

# In[18]:


group_cols = [
    "conc_unit",
    "species_group",
    "endpoint",
    "effect",
    "fragment_bin",
    "has_metal_group",
]

# train_df["actual_log10c"] = [g.y.item() * target_std + target_mean for g in train_dataset]
train_df["actual_log10c"] = [g.y.item() for g in train_dataset]

group_summaries = {}

for category in categorical_cols:

    print(category)

    plot_group_training(
        history,
        record_categories=[category],
        metric="loss",   # or "mae" / "rmse"
        top_n=4,
        label_encoder=categorical_encoder,
    )
    
    summary = summarize_by_group(results_df, train_df, category, min_count=25)

    plot_group_mae(
        summary, 
        category=category,
    )


# In[ ]:




