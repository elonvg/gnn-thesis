from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaning import (  # noqa: E402
    fragment_count,
    has_metal,
    is_salt,
    is_single_node,
    mask_data,
    preprocess,
    print_mol_types,
)
from src.data.simple_featurizer import simple_featurizer  # noqa: E402
from src.data.graph_building import build_graph_features  # noqa: E402
from src.data.io import load_data  # noqa: E402
from src.data.metadata import build_config, sequential_encoder  # noqa: E402
from src.data.sampling import LoadData  # noqa: E402
from src.data.splitting import _build_dataset, load_butina_clusters  # noqa: E402
from src.models.afp_flex import AFPFlex  # noqa: E402
from src.models.meta_encoder import MetaEncoder, TaxonomyOneHot  # noqa: E402
from src.models.toxicity_model import ToxicityModel  # noqa: E402
from src.training.loops import train  # noqa: E402


SELECTED_COLUMNS = [
    "SK_unique_id",
    "species_common_name",
    "species_latin_name",
    "CAS",
    "chemical_name",
    "conc_unit",
    "conc",
    "duration",
    "duration_unit",
    "effect",
    "endpoint",
    "SMILES",
    "organism_lifestage_categorized",
    "administration_route_categorized",
    "NCBI_sci_name",
    "NCBI_last_known_rank",
    "NCBI_rank_superkingdom",
    "NCBI_rank_kingdom",
    "NCBI_rank_phylum",
    "NCBI_rank_subphylum",
    "NCBI_rank_class",
    "NCBI_rank_order",
    "NCBI_rank_family",
    "NCBI_rank_genus",
    "NCBI_rank_species",
    "species_group_corrected",
]

RENAME_COLUMNS = {
    "species_group_corrected": "species_group",
    "organism_lifestage_categorized": "organism_lifestage",
    "administration_route_categorized": "administration_route",
    "NCBI_rank_superkingdom": "superkingdom",
    "NCBI_rank_kingdom": "kingdom",
    "NCBI_rank_phylum": "phylum",
    "NCBI_rank_subphylum": "subphylum",
    "NCBI_rank_class": "class",
    "NCBI_rank_order": "order",
    "NCBI_rank_family": "family",
    "NCBI_rank_genus": "genus",
    "NCBI_rank_species": "species",
    "NCBI_sci_name": "species_sci_name",
    "NCBI_last_known_rank": "taxid",
}

FILTERS = {
    "duration_unit": ["h"],
    "effect": ["MOR", "POP", "GRO", "BEH", "REP", "ITX", "PHY", "DVP", "MPH"],
}

TAXONOMY_COLS = ("class", "family", "genus", "species")
TAX_EMBEDDING = {"taxid": 16}
CATEGORICAL_COLS = [
    "species_group",
    "conc_unit",
    "endpoint",
    "effect",
    "is_salt",
    "has_metal",
    "is_single_node",
]
NUMERICAL_COLS = ["duration", "fragment_count"]
RECORD_CATEGORIES = ["species_group", "endpoint", "effect", "conc_unit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one GroupKFold fold on one GPU/process.")
    parser.add_argument("--fold-id", type=int, required=True, help="Fold index to train, zero-based.")
    parser.add_argument("--folds", type=int, default=5, help="Total number of GroupKFold folds.")
    parser.add_argument("--random-state", type=int, default=11)
    parser.add_argument("--max-rows", type=int, default=6000, help="Use <= 0 to disable row limiting.")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "Data" / "toxicity_all.csv")
    parser.add_argument(
        "--cluster-csv-path",
        type=Path,
        default=PROJECT_ROOT / "Data" / "moredata" / "original" / "butina_cluster_lookup.csv",
    )
    parser.add_argument("--cluster-col", default="Cluster_at_cutoff_0.2")
    parser.add_argument(
        "--pretrained-taxid-path",
        type=Path,
        default=PROJECT_ROOT / "Data" / "moredata" / "pretrained_tax_emb.pkl.zip",
    )
    parser.add_argument("--use-pretrained-taxid", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sampling-attribute", default="species_group")

    parser.add_argument("--tax-dim", type=int, default=16)
    parser.add_argument("--pretrained-tax-dim", type=int, default=768)
    parser.add_argument("--pretrained-taxid-output-dim", type=int, default=128)
    parser.add_argument("--categorical-dim", type=int, default=16)
    parser.add_argument("--numeric-dim", type=int, default=16)
    parser.add_argument("--meta-dropout", type=float, default=0.3)

    parser.add_argument("--gnn-hidden-dim", type=int, default=64)
    parser.add_argument("--gnn-out-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-timesteps", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--final-hidden-dim", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-beta", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="gnn-thesis")
    parser.add_argument("--wandb-entity", default="elonvg-chalmers-university-of-technology")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)

    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "outputs" / "models")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_max_rows(max_rows: int) -> int | None:
    return None if max_rows is None or max_rows <= 0 else max_rows


def load_and_filter_data(args: argparse.Namespace) -> pd.DataFrame:
    df_all = load_data(args.data_path, SELECTED_COLUMNS)
    df_all = df_all.rename(columns=RENAME_COLUMNS)

    df_all["organism_lifestage"] = df_all["organism_lifestage"].fillna("adult")
    df_all["administration_route"] = df_all["administration_route"].fillna("fill")
    df_all["duration_unit"] = df_all["duration_unit"].fillna("h")

    mask = mask_data(
        df_all,
        filters=FILTERS,
        require_duration=False,
        require_taxonomy=True,
        taxonomy_columns=TAXONOMY_COLS,
    )
    df_filtered = df_all.loc[mask].copy()

    for col in TAXONOMY_COLS:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce").astype("Int64")

    print("Loaded and filtered training data", flush=True)
    print(f"Rows in full data: {len(df_all):,}", flush=True)
    print(f"Rows after filter: {len(df_filtered):,}", flush=True)
    return df_filtered


def preprocess_data(df_filtered: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    max_rows = normalize_max_rows(args.max_rows)
    if max_rows is not None and len(df_filtered) > max_rows:
        df_filtered = df_filtered.sample(n=max_rows, random_state=args.random_state).reset_index(drop=True)
    else:
        df_filtered = df_filtered.reset_index(drop=True)

    df_processed = preprocess(
        df_filtered.copy(),
        split_salts=False,
        remove_lone=False,
        remove_metals=False,
        max_conc_value=10000,
        duration_fill_value=1e-6,
        max_duration_hours=9000.0,
        log_transform_duration=True,
        keep_duration_raw=True,
    )

    print()
    print(f"Rows before preprocessing: {len(df_filtered):,}", flush=True)
    print(f"Rows after preprocessing:  {len(df_processed):,}", flush=True)
    print(f"Rows removed: {len(df_filtered) - len(df_processed):,}", flush=True)
    print_mol_types(df_processed)
    return df_processed


def build_feature_frame(df_processed: pd.DataFrame, args: argparse.Namespace):
    df_processed = df_processed.copy()
    df_processed["features"] = df_processed["SMILES"].apply(simple_featurizer)

    model_tax_embedding = (
        {key: value for key, value in TAX_EMBEDDING.items() if key != "taxid"}
        if args.use_pretrained_taxid
        else TAX_EMBEDDING
    )
    df_tax = df_processed[list(TAX_EMBEDDING.keys())].copy()
    df_tax, tax_encoders = sequential_encoder(df_tax, TAX_EMBEDDING.keys())
    config_tax = build_config(df_tax, model_tax_embedding)

    df_processed["fragment_count"] = df_processed["SMILES"].apply(fragment_count).astype(float)
    df_processed["is_salt"] = df_processed["SMILES"].apply(is_salt).astype(float)
    df_processed["has_metal"] = df_processed["SMILES"].apply(has_metal).astype(float)
    df_processed["is_single_node"] = df_processed["SMILES"].apply(is_single_node).astype(float)

    df_categorical = df_processed[CATEGORICAL_COLS].copy()
    df_categorical, categorical_encoder = sequential_encoder(df_categorical, CATEGORICAL_COLS)
    config_categorical = build_config(df_categorical, CATEGORICAL_COLS)

    features = build_graph_features(
        df_processed,
        df_tax,
        TAX_EMBEDDING,
        df_categorical,
        CATEGORICAL_COLS,
        NUMERICAL_COLS,
    )

    print()
    print(f"{len(df_processed):,} rows with graph features created", flush=True)
    print(f"Graph objects created: {len(features):,}", flush=True)
    print("Categorical embedding config:", flush=True)
    print(config_categorical, flush=True)
    print("Numerical encoding for:", NUMERICAL_COLS, flush=True)
    if args.use_pretrained_taxid:
        print(f"Using pretrained taxid embeddings: {args.pretrained_taxid_path}", flush=True)

    return features, config_tax, categorical_encoder, config_categorical


def butina_group_key(smiles: str, smiles_to_cluster: dict) -> str:
    cluster_id = smiles_to_cluster.get(smiles, np.nan)
    if pd.isna(cluster_id):
        return f"__missing__::{smiles}"
    if isinstance(cluster_id, float) and cluster_id.is_integer():
        cluster_id = int(cluster_id)
    return f"cluster::{cluster_id}"


def make_group_kfold_split(features, args: argparse.Namespace):
    smiles_to_cluster = load_butina_clusters(args.cluster_csv_path, args.cluster_col)
    groups = pd.Series(
        [butina_group_key(graph.smiles, smiles_to_cluster) for graph in features],
        name="butina_group",
    )

    if args.fold_id < 0 or args.fold_id >= args.folds:
        raise ValueError(f"--fold-id must be between 0 and {args.folds - 1}; got {args.fold_id}.")

    group_kfold = GroupKFold(n_splits=args.folds)
    splits = list(group_kfold.split(features, groups=groups))
    train_idx, val_idx = splits[args.fold_id]

    train_groups = groups.iloc[train_idx]
    val_groups = groups.iloc[val_idx]

    print()
    print("GroupKFold split", flush=True)
    print(f"Fold: {args.fold_id}/{args.folds - 1}", flush=True)
    print(f"Rows: {len(groups):,}", flush=True)
    print(f"Unique groups: {groups.nunique():,}", flush=True)
    print(f"Missing group values: {groups.isna().sum():,}", flush=True)
    print(f"Fallback missing-cluster rows: {groups.str.startswith('__missing__::').sum():,}", flush=True)
    print(f"Train rows: {len(train_idx):,}; val rows: {len(val_idx):,}", flush=True)
    print(f"Train groups: {train_groups.nunique():,}; val groups: {val_groups.nunique():,}", flush=True)
    print(f"Shared train/val groups: {len(set(train_groups) & set(val_groups)):,}", flush=True)

    return train_idx, val_idx, groups


def print_split_info(train_dataset, val_dataset) -> None:
    train_targets = np.array([g.y.item() for g in train_dataset])
    val_targets = np.array([g.y.item() for g in val_dataset])
    train_smiles = [g.smiles for g in train_dataset]
    val_smiles = [g.smiles for g in val_dataset]
    total_len = len(train_dataset) + len(val_dataset)

    print()
    print(f"Train size: {len(train_dataset):,} ({len(train_dataset) / total_len:.1%})", flush=True)
    print(f"Val size:   {len(val_dataset):,} ({len(val_dataset) / total_len:.1%})", flush=True)
    print(f"Unique molecules in train: {len(set(train_smiles)):,}", flush=True)
    print(f"Unique molecules in val:   {len(set(val_smiles)):,}", flush=True)
    print(f"Val molecules not in train: {len(set(val_smiles) - set(train_smiles)):,}", flush=True)
    print("Target distribution", flush=True)
    print(f"Train mean/std: {train_targets.mean():.4f} / {train_targets.std():.4f}", flush=True)
    print(f"Val mean/std:   {val_targets.mean():.4f} / {val_targets.std():.4f}", flush=True)


def create_loaders(train_dataset, val_dataset, args: argparse.Namespace):
    train_loader = LoadData(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        attribute=args.sampling_attribute,
    )
    val_loader = LoadData(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        attribute=args.sampling_attribute,
        target_dataset=train_dataset,
    )

    print()
    print(f"Train loader: {len(train_dataset):,} samples, {len(train_loader):,} batches", flush=True)
    print(f"Val loader:   {len(val_dataset):,} samples, {len(val_loader):,} batches", flush=True)
    return train_loader, val_loader


def build_model(features, config_tax, config_categorical, args: argparse.Namespace, device: torch.device):
    atom_feature_dim = features[0].x.shape[1]
    edge_feature_dim = features[0].edge_attr.shape[1]

    meta_encoder = MetaEncoder(
        taxonomy_encoder_cls=TaxonomyOneHot,
        config_tax=config_tax,
        tax_output_dim=args.tax_dim,
        pretrained_taxid_path=args.pretrained_taxid_path if args.use_pretrained_taxid else None,
        pretrained_tax_dim=args.pretrained_tax_dim,
        pretrained_taxid_output_dim=args.pretrained_taxid_output_dim,
        pretrained_taxid_encoder_kwargs={},
        config_categorical=config_categorical,
        categorical_output_dim=args.categorical_dim,
        numerical_columns=NUMERICAL_COLS,
        numeric_output_dim=args.numeric_dim,
        dropout=args.meta_dropout,
    ).to(device)

    model_gnn = AFPFlex(
        in_channels=atom_feature_dim,
        edge_dim=edge_feature_dim,
        hidden_channels=args.gnn_hidden_dim,
        out_channels=args.gnn_out_dim,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        dropout=args.dropout,
    ).to(device)

    model = ToxicityModel(
        model_gnn,
        meta_encoder,
        hidden_dim=args.final_hidden_dim,
    ).to(device)

    n_params_meta = sum(p.numel() for p in meta_encoder.parameters() if p.requires_grad)
    n_params_gnn = sum(p.numel() for p in model_gnn.parameters() if p.requires_grad)
    n_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info = {
        "atom_feature_dim": atom_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "n_params_meta": n_params_meta,
        "n_params_gnn": n_params_gnn,
        "n_params_total": n_params_total,
        "gnn_name": type(model_gnn).__name__,
    }

    print()
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Meta encoder trainable parameters: {n_params_meta:,}", flush=True)
    print(f"GNN trainable parameters: {n_params_gnn:,}", flush=True)
    print(f"Total trainable parameters: {n_params_total:,}", flush=True)
    return model, model_info


def serializable_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def init_wandb(args: argparse.Namespace, model_info: dict, train_dataset, val_dataset, groups):
    if not args.wandb or wandb is None:
        print()
        print("wandb disabled or not installed; running without experiment tracking.", flush=True)
        return None

    group_name = args.wandb_group or f"{model_info['gnn_name']}-groupkfold-{args.random_state}"
    run_name = args.wandb_name or f"{model_info['gnn_name']}-fold-{args.fold_id}"
    wandb_dir = PROJECT_ROOT / "outputs" / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        dir=str(wandb_dir),
        job_type="train",
        group=group_name,
        name=run_name,
        tags=["script", "group_kfold", model_info["gnn_name"]],
        config={
            **serializable_args(args),
            "filters": FILTERS,
            "tax_embedding": TAX_EMBEDDING,
            "categorical_cols": CATEGORICAL_COLS,
            "numerical_cols": NUMERICAL_COLS,
            "record_categories": RECORD_CATEGORIES,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "train_groups": groups.iloc[[g.row_id.item() for g in train_dataset]].nunique(),
            "val_groups": groups.iloc[[g.row_id.item() for g in val_dataset]].nunique(),
            **model_info,
        },
    )
    run.define_metric("epoch")
    for metric_prefix in ("train/*", "val/*", "optimizer/*"):
        run.define_metric(metric_prefix, step_metric="epoch")
    return run


def save_checkpoint(args: argparse.Namespace, model, model_info: dict, history: dict) -> Path | None:
    if not args.save_checkpoint:
        return None

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"{model_info['gnn_name']}_fold_{args.fold_id}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_info": model_info,
            "args": serializable_args(args),
            "history_all": history["history_all"],
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    print("Setup complete", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Data file: {args.data_path}", flush=True)
    print(f"Fold id: {args.fold_id}", flush=True)

    df_filtered = load_and_filter_data(args)
    df_processed = preprocess_data(df_filtered, args)
    features, config_tax, categorical_encoder, config_categorical = build_feature_frame(df_processed, args)

    train_idx, val_idx, groups = make_group_kfold_split(features, args)
    train_dataset = _build_dataset(features, train_idx)
    val_dataset = _build_dataset(features, val_idx)
    print_split_info(train_dataset, val_dataset)

    train_loader, val_loader = create_loaders(train_dataset, val_dataset, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_info = build_model(features, config_tax, config_categorical, args, device)

    loss_fn = torch.nn.SmoothL1Loss(beta=args.loss_beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        min_lr=args.scheduler_min_lr,
    )

    wandb_run = init_wandb(args, model_info, train_dataset, val_dataset, groups)

    print()
    print("Training configuration", flush=True)
    print(f"epochs = {args.epochs}", flush=True)
    print(f"learning_rate = {args.learning_rate}", flush=True)
    print(f"weight_decay = {args.weight_decay}", flush=True)
    print(f"loss = {loss_fn.__class__.__name__}", flush=True)
    print(f"early_stopping_patience = {args.early_stopping_patience}", flush=True)

    try:
        model, history = train(
            model,
            train_loader,
            val_loader=val_loader,
            test_loader=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            device=device,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            record_categories=RECORD_CATEGORIES,
            record_joint_categories=("endpoint", "species_group"),
            label_encoder=categorical_encoder,
            run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    checkpoint_path = save_checkpoint(args, model, model_info, history)

    print()
    print("Training finished", flush=True)
    print(f"Epochs ran: {history['history_all']['epochs_ran']}", flush=True)
    print(f"Best epoch: {history['history_all']['best_epoch']}", flush=True)
    print(f"Best monitor value: {history['history_all']['best_monitor_value']}", flush=True)
    if checkpoint_path is not None:
        print(f"Saved checkpoint: {checkpoint_path}", flush=True)

    summary = {
        "fold_id": args.fold_id,
        "epochs_ran": history["history_all"]["epochs_ran"],
        "best_epoch": history["history_all"]["best_epoch"],
        "best_monitor_value": history["history_all"]["best_monitor_value"],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
