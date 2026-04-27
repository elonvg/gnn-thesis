from pathlib import Path
import sys
import time

_startup_time = time.time()
print("Importing core scientific packages...", flush=True)
import numpy as np
import pandas as pd

print("Importing torch...", flush=True)
import torch
print(f"Imported torch {torch.__version__} in {time.time() - _startup_time:.1f}s", flush=True)

print("Importing sklearn and optional experiment tracking...", flush=True)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

"GNN Tox training script"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "toxicity_all.csv"
PRETRAINED_TAXID_PATH = PROJECT_ROOT / "Data" / "moredata" / "pretrained_tax_emb.pkl.zip"
CLUSTER_CSV_PATH = PROJECT_ROOT / "Data" / "moredata" / "original" / "butina_cluster_lookup.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"

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

REQUIRE_DURATION = False
REQUIRE_TAXONOMY = True

MAX_ROWS = 80000
RANDOM_STATE = 11
SET_GLOBAL_SEED = True

SPLIT_SALTS = False
REMOVE_LONE = False
REMOVE_METALS = False
MAX_CONC_VALUE = 10000
DURATION_FILL_VALUE = 1e-6
MAX_DURATION_HOURS = 9000.0
LOG_TRANSFORM_DURATION = True
KEEP_DURATION_RAW = True

USE_PRETRAINED_TAXID = True
TAX_EMBEDDING = {
    "taxid": 16,
}

CATEGORICAL_COLS = [
    "species_group",
    "conc_unit",
    "endpoint",
    "effect",
    "is_salt",
    "has_metal",
    "is_single_node",
]
NUMERICAL_COLS = [
    "duration",
    "fragment_count",
]

SPLIT_METHOD = "butina"
CLUSTER_COL = "Cluster_at_cutoff_0.2"
STRATIFY_BY = ["species_group", "endpoint", "effect", "conc_unit"]
FRAC_TRAIN = 0.7
FRAC_VALID = 0.1
FRAC_TEST = 0.2

BATCH_SIZE = 256
SAMPLING_ATTRIBUTE = "species_group"

TAX_DIM = 32
PRETRAINED_TAX_DIM = 768
PRETRAINED_TAXID_OUTPUT_DIM = 128
CATEGORICAL_DIM = 32
NUMERIC_DIM = 32
META_DROPOUT = 0.3

GNN_HIDDEN_DIM = 128
GNN_OUT_DIM = 128
NUM_LAYERS = 3
NUM_TIMESTEPS = 2
DROPOUT = 0.3
FINAL_HIDDEN_DIM = 128

EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LOSS_BETA = 0.5
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_MIN_DELTA = 1e-4
RECORD_CATEGORIES = CATEGORICAL_COLS

ENABLE_WANDB = True
# ENABLE_WANDB = False
WANDB_PROJECT = "gnn-thesis"
WANDB_ENTITY = "elonvg-chalmers-university-of-technology"
WANDB_JOB_TYPE = "train"

GROUP_SUMMARY_MIN_COUNT = 25
LARGEST_ERRORS_N = 10
SAVE_RESULTS = True
RESULTS_CSV_PATH = OUTPUT_DIR / "model_training_results.csv"
LARGEST_ERRORS_CSV_PATH = OUTPUT_DIR / "model_training_largest_errors.csv"


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
from src.data.featurization import simple_featurizer  # noqa: E402
from src.data.graph_building import build_graph_features  # noqa: E402
from src.data.io import load_data  # noqa: E402
from src.data.metadata import build_config, sequential_encoder  # noqa: E402
from src.data.sampling import LoadData  # noqa: E402
from src.data.splitting import butina_split  # noqa: E402
from src.models.afp_flex import AFPFlex  # noqa: E402
from src.models.meta_encoder import MetaEncoder, TaxonomyOneHot  # noqa: E402
from src.models.toxicity_model import ToxicityModel  # noqa: E402
from src.training.loops import predict_df, train  # noqa: E402


RESULT_TARGET_COL = "actual_log10c"
PREDICTION_COL = "pred_log10c"
TRAIN_TARGET_CANDIDATES = ("actual_log10c", "log10c")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_columns(frame, columns, frame_name):
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"{frame_name} is missing required column(s): {missing_str}")


def normalize_group_labels(series):
    labels = series.astype("string")
    labels = labels.fillna("Missing").replace({"<NA>": "Missing", "nan": "Missing", "None": "Missing"})
    return labels.astype(str)


def resolve_train_target_col(train_df):
    for candidate in TRAIN_TARGET_CANDIDATES:
        if candidate in train_df.columns:
            return candidate

    expected = ", ".join(TRAIN_TARGET_CANDIDATES)
    raise KeyError(f"train_df must include one of the training target columns: {expected}")


def summarize_by_group(results_df, train_df, group_col, min_count=20):
    require_columns(results_df, [group_col, RESULT_TARGET_COL, PREDICTION_COL], "results_df")
    require_columns(train_df, [group_col], "train_df")

    train_target_col = resolve_train_target_col(train_df)

    df = results_df.copy()
    df = df.dropna(subset=[RESULT_TARGET_COL, PREDICTION_COL]).copy()
    df["group"] = normalize_group_labels(df[group_col])

    train_groups = train_df.copy()
    train_groups = train_groups.dropna(subset=[train_target_col]).copy()
    train_groups["group"] = normalize_group_labels(train_groups[group_col])

    global_mean = train_groups[train_target_col].mean()
    if np.isnan(global_mean):
        raise ValueError("train_df does not contain any non-null training target values.")

    baseline_stats = (
        train_groups.groupby("group")[train_target_col]
        .agg(train_n="size", baseline_log10c="mean")
        .reset_index()
    )

    df = df.merge(baseline_stats, on="group", how="left")
    df["baseline_log10c"] = df["baseline_log10c"].fillna(global_mean)
    df["train_n"] = df["train_n"].fillna(0).astype(int)
    df["baseline_source"] = np.where(df["train_n"] > 0, "train_group_mean", "global_train_mean")

    df["model_abs_error"] = (df[PREDICTION_COL] - df[RESULT_TARGET_COL]).abs()
    df["baseline_abs_error"] = (df["baseline_log10c"] - df[RESULT_TARGET_COL]).abs()
    df["model_sq_error"] = (df[PREDICTION_COL] - df[RESULT_TARGET_COL]) ** 2
    df["baseline_sq_error"] = (df["baseline_log10c"] - df[RESULT_TARGET_COL]) ** 2

    summary = (
        df.groupby("group")
        .agg(
            n=("group", "size"),
            train_n=("train_n", "first"),
            baseline_source=("baseline_source", "first"),
            baseline_log10c=("baseline_log10c", "first"),
            model_mae=("model_abs_error", "mean"),
            baseline_mae=("baseline_abs_error", "mean"),
            model_rmse=("model_sq_error", lambda x: np.sqrt(x.mean())),
            baseline_rmse=("baseline_sq_error", lambda x: np.sqrt(x.mean())),
        )
        .reset_index()
    )

    summary["mae_gain"] = summary["baseline_mae"] - summary["model_mae"]
    summary["rmse_gain"] = summary["baseline_rmse"] - summary["model_rmse"]

    if min_count is not None:
        summary = summary[summary["n"] >= min_count]

    return summary.sort_values(["n", "mae_gain"], ascending=[False, False]).reset_index(drop=True)


def load_and_filter_data():
    df_all = load_data(DATA_PATH, SELECTED_COLUMNS)
    df_all = df_all.rename(columns=RENAME_COLUMNS)

    df_all["organism_lifestage"] = df_all["organism_lifestage"].fillna("adult")
    df_all["administration_route"] = df_all["administration_route"].fillna("fill")
    df_all["duration_unit"] = df_all["duration_unit"].fillna("h")

    mask = mask_data(
        df_all,
        filters=FILTERS,
        require_duration=REQUIRE_DURATION,
        require_taxonomy=REQUIRE_TAXONOMY
    )
    df_filtered = df_all.loc[mask].copy()


    print("Loaded and filtered training data")
    print(f"Rows in full data: {len(df_all):,}")
    print(f"Rows after filter: {len(df_filtered):,}")

    return df_filtered


def preprocess_data(df_filtered):
    if MAX_ROWS is not None and len(df_filtered) > MAX_ROWS:
        df_filtered = df_filtered.sample(n=MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        df_filtered = df_filtered.reset_index(drop=True)

    df_processed = preprocess(
        df_filtered.copy(),
        split_salts=SPLIT_SALTS,
        remove_lone=REMOVE_LONE,
        remove_metals=REMOVE_METALS,
        max_conc_value=MAX_CONC_VALUE,
        duration_fill_value=DURATION_FILL_VALUE,
        max_duration_hours=MAX_DURATION_HOURS,
        log_transform_duration=LOG_TRANSFORM_DURATION,
        keep_duration_raw=KEEP_DURATION_RAW,
    )

    print()
    print(f"Rows before preprocessing: {len(df_filtered):,}")
    print(f"Rows after preprocessing:  {len(df_processed):,}")
    print(f"Rows removed: {len(df_filtered) - len(df_processed):,}")
    print_mol_types(df_processed)

    return df_processed


def add_graph_features(df_processed):
    df_processed = df_processed.copy()
    df_processed["features"] = df_processed["SMILES"].apply(simple_featurizer)
    print()
    print(f"{len(df_processed):,} rows with graph features created")
    return df_processed


def build_metadata(df_processed):
    model_tax_embedding = (
        {key: value for key, value in TAX_EMBEDDING.items() if key != "taxid"}
        if USE_PRETRAINED_TAXID
        else TAX_EMBEDDING
    )

    df_tax = df_processed[list(TAX_EMBEDDING.keys())].copy()
    df_tax, tax_encoders = sequential_encoder(df_tax, TAX_EMBEDDING.keys())
    config_tax = build_config(df_tax, model_tax_embedding)

    if USE_PRETRAINED_TAXID:
        print()
        print("Using pretrained taxid embeddings")
        print(f"Pretrained taxid path: {PRETRAINED_TAXID_PATH}")
    else:
        print()
        print("Taxonomy embedding config")
        print(config_tax)

    df_processed = df_processed.copy()
    df_processed["fragment_count"] = df_processed["SMILES"].apply(fragment_count).astype(float)
    df_processed["is_salt"] = df_processed["SMILES"].apply(is_salt).astype(float)
    df_processed["has_metal"] = df_processed["SMILES"].apply(has_metal).astype(float)
    df_processed["is_single_node"] = df_processed["SMILES"].apply(is_single_node).astype(float)

    df_categorical = df_processed[CATEGORICAL_COLS].copy()
    df_categorical, categorical_encoder = sequential_encoder(df_categorical, CATEGORICAL_COLS)
    config_categorical = build_config(df_categorical, CATEGORICAL_COLS)

    print()
    print("Categorical embedding config:")
    print(config_categorical)
    print("Numerical encoding for:")
    print(NUMERICAL_COLS)

    return df_processed, df_tax, config_tax, df_categorical, categorical_encoder, config_categorical


def print_split_info(train_dataset, val_dataset, test_dataset):
    train_targets = np.array([g.y.item() for g in train_dataset])
    val_targets = np.array([g.y.item() for g in val_dataset])
    test_targets = np.array([g.y.item() for g in test_dataset])
    train_smiles = [g.smiles for g in train_dataset]
    val_smiles = [g.smiles for g in val_dataset]
    test_smiles = [g.smiles for g in test_dataset]
    total_len = len(train_dataset) + len(val_dataset) + len(test_dataset)

    print()
    print(f"Train size: {len(train_dataset):,} ({len(train_dataset) / total_len:.1%})")
    print(f"Val size:   {len(val_dataset):,} ({len(val_dataset) / total_len:.1%})")
    print(f"Test size:  {len(test_dataset):,} ({len(test_dataset) / total_len:.1%})")
    print(f"Unique molecules in train: {len(set(train_smiles)):,}")
    print(f"Unique molecules in val:   {len(set(val_smiles)):,}")
    print(f"Unique molecules in test:  {len(set(test_smiles)):,}")
    print(f"Val molecules not in train:  {len(set(val_smiles) - set(train_smiles)):,}")
    print(f"Test molecules not in train: {len(set(test_smiles) - set(train_smiles)):,}")
    print("Target distribution")
    print(f"Train mean/std: {train_targets.mean():.4f} / {train_targets.std():.4f}")
    print(f"Val mean/std:   {val_targets.mean():.4f} / {val_targets.std():.4f}")
    print(f"Test mean/std:  {test_targets.mean():.4f} / {test_targets.std():.4f}")


def print_loader_info(train_loader, val_loader, test_loader):
    loaders = {
        "Train": train_loader,
        "Val": val_loader,
        "Test": test_loader,
    }

    print()
    for loader_name, loader in loaders.items():
        dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else "unknown"
        sampled_size = getattr(getattr(loader, "sampler", None), "num_samples", dataset_size)
        batch_size = getattr(loader, "batch_size", "unknown")
        print(
            f"{loader_name}: {dataset_size} dataset samples, "
            f"{sampled_size} sampled samples, {len(loader)} batches "
            f"(batch_size={batch_size})"
        )


def create_loaders(train_dataset, val_dataset, test_dataset):
    train_loader = LoadData(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        attribute=SAMPLING_ATTRIBUTE,
    )
    val_loader = LoadData(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        attribute=SAMPLING_ATTRIBUTE,
        target_dataset=train_dataset,
    )
    test_loader = LoadData(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        attribute=SAMPLING_ATTRIBUTE,
        target_dataset=train_dataset,
    )
    print_loader_info(train_loader, val_loader, test_loader)
    return train_loader, val_loader, test_loader


def build_model(features, config_tax, config_categorical):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atom_feature_dim = features[0].x.shape[1]
    edge_feature_dim = features[0].edge_attr.shape[1]

    meta_encoder = MetaEncoder(
        taxonomy_encoder_cls=TaxonomyOneHot,
        config_tax=config_tax,
        tax_output_dim=TAX_DIM,
        pretrained_taxid_path=PRETRAINED_TAXID_PATH if USE_PRETRAINED_TAXID else None,
        pretrained_tax_dim=PRETRAINED_TAX_DIM,
        pretrained_taxid_output_dim=PRETRAINED_TAXID_OUTPUT_DIM,
        config_categorical=config_categorical,
        categorical_output_dim=CATEGORICAL_DIM,
        numerical_columns=NUMERICAL_COLS,
        numeric_output_dim=NUMERIC_DIM,
        dropout=META_DROPOUT,
    ).to(device)

    model_gnn = AFPFlex(
        in_channels=atom_feature_dim,
        edge_dim=edge_feature_dim,
        hidden_channels=GNN_HIDDEN_DIM,
        out_channels=GNN_OUT_DIM,
        num_layers=NUM_LAYERS,
        num_timesteps=NUM_TIMESTEPS,
        dropout=DROPOUT,
    ).to(device)

    model = ToxicityModel(
        model_gnn,
        meta_encoder,
        hidden_dim=FINAL_HIDDEN_DIM,
    ).to(device)

    n_params_meta = sum(p.numel() for p in meta_encoder.parameters() if p.requires_grad)
    n_params_gnn = sum(p.numel() for p in model_gnn.parameters() if p.requires_grad)
    n_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()
    print(f"Device: {device}")
    print(f"Atom feature dim: {atom_feature_dim}")
    print(f"Edge feature dim: {edge_feature_dim}")
    print(f"Meta encoder trainable parameters: {n_params_meta:,}")
    print(f"GNN trainable parameters: {n_params_gnn:,}")
    print(f"Total trainable parameters: {n_params_total:,}")
    print(model)

    model_info = {
        "atom_feature_dim": atom_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "n_params_meta": n_params_meta,
        "n_params_gnn": n_params_gnn,
        "n_params_total": n_params_total,
        "gnn_name": type(model_gnn).__name__,
    }
    return model, device, model_info


def init_wandb(model_info):
    if not ENABLE_WANDB or wandb is None:
        print()
        print("wandb disabled or not installed; running without experiment tracking.")
        return None

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type=WANDB_JOB_TYPE,
        tags=["script", SPLIT_METHOD, model_info["gnn_name"]],
        config={
            "random_state": RANDOM_STATE,
            "n_samples": MAX_ROWS,
            "filters": FILTERS,
            "split_salt": SPLIT_SALTS,
            "remove_lone": REMOVE_LONE,
            "remove_metals": REMOVE_METALS,
            "max_conc_value": MAX_CONC_VALUE,
            "duration_fill_value": DURATION_FILL_VALUE,
            "max_duration_hours": MAX_DURATION_HOURS,
            "log_transform_duration": LOG_TRANSFORM_DURATION,
            "num_atom_features": model_info["atom_feature_dim"],
            "num_bond_features": model_info["edge_feature_dim"],
            "tax_embedding": TAX_EMBEDDING,
            "use_pretrained_taxid": USE_PRETRAINED_TAXID,
            "categorical_cols": CATEGORICAL_COLS,
            "numerical_cols": NUMERICAL_COLS,
            "split_method": SPLIT_METHOD,
            "butina_cluster_col": CLUSTER_COL,
            "stratify_by": STRATIFY_BY,
            "frac_train": FRAC_TRAIN,
            "frac_valid": FRAC_VALID,
            "frac_test": FRAC_TEST,
            "batch_size": BATCH_SIZE,
            "taxonomy_encoder": TaxonomyOneHot.__name__,
            "gnn_model": model_info["gnn_name"],
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
            "n_params_meta": model_info["n_params_meta"],
            "n_params_gnn": model_info["n_params_gnn"],
            "n_params_total": model_info["n_params_total"],
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "SmoothL1Loss",
            "loss_beta": LOSS_BETA,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        },
    )
    run.define_metric("epoch")
    for metric_prefix in ("train/*", "val/*", "test/*", "optimizer/*"):
        run.define_metric(metric_prefix, step_metric="epoch")

    return run


def train_model(model, train_loader, val_loader, test_loader, device, model_info, categorical_encoder):
    loss_fn = torch.nn.SmoothL1Loss(beta=LOSS_BETA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
        min_lr=SCHEDULER_MIN_LR,
    )
    wandb_run = init_wandb(model_info)

    print()
    print("Training configuration")
    print(f"epochs = {EPOCHS}")
    print(f"learning_rate = {LEARNING_RATE}")
    print(f"weight_decay = {WEIGHT_DECAY}")
    print(f"loss = {loss_fn.__class__.__name__}")
    print(f"early_stopping_patience = {EARLY_STOPPING_PATIENCE}")

    try:
        model_trained, history = train(
            model,
            train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=EPOCHS,
            device=device,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
            record_categories=RECORD_CATEGORIES,
            label_encoder=categorical_encoder,
            run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return model_trained, history


def build_analysis_frames(df_processed, train_dataset):
    analysis_df = df_processed.reset_index(drop=True)[
        [
            "species_group",
            "species_latin_name",
            "endpoint",
            "effect",
            "fragment_count",
            "has_metal",
            "is_single_node",
            "is_salt",
            "conc_unit",
        ]
    ].copy()

    analysis_df["fragment_bin"] = pd.cut(
        analysis_df["fragment_count"],
        [-1, 1, 2, np.inf],
        labels=["1", "2", "3+"],
    )
    analysis_df["has_metal_group"] = analysis_df["has_metal"].map({0.0: "No metal", 1.0: "Has metal"})
    analysis_df["is_single_group"] = analysis_df["is_single_node"].map(
        {0.0: "Not single-node", 1.0: "Single-node"}
    )
    analysis_df["is_salt"] = analysis_df["is_salt"].map({0.0: "Not salt", 1.0: "Is salt"})

    train_df = analysis_df.iloc[[g.row_id.item() for g in train_dataset]].copy()
    train_df["actual_log10c"] = [g.y.item() for g in train_dataset]

    return analysis_df, train_df


def evaluate_results(model, test_loader, device, df_processed, train_dataset):
    analysis_df, train_df = build_analysis_frames(df_processed, train_dataset)

    results_df = predict_df(model, test_loader, device, cols=["row_id", "smiles", "taxid_raw"])
    results_df["row_id"] = results_df["row_id"].astype(int)
    results_df["taxid"] = results_df["taxid_raw"].astype(int)
    results_df = results_df.drop(columns="taxid_raw").join(analysis_df, on="row_id")

    results_df["pred_log10c"] = results_df["pred_norm"]
    results_df["actual_log10c"] = results_df["actual_norm"]
    results_df["residual_log10c"] = results_df["pred_log10c"] - results_df["actual_log10c"]
    results_df["abs_error_log10c"] = results_df["residual_log10c"].abs()
    results_df["pred_conc"] = 10 ** results_df["pred_log10c"]
    results_df["actual_conc"] = 10 ** results_df["actual_log10c"]
    results_df["fold_error"] = np.maximum(
        results_df["pred_conc"] / results_df["actual_conc"],
        results_df["actual_conc"] / results_df["pred_conc"],
    )

    summary_metrics = {
        "test/r2_norm": r2_score(results_df["actual_norm"], results_df["pred_norm"]),
        "test/r2_log10c": r2_score(results_df["actual_log10c"], results_df["pred_log10c"]),
        "test/rmse_log10c": mean_squared_error(results_df["actual_log10c"], results_df["pred_log10c"]) ** 0.5,
        "test/mae_log10c": mean_absolute_error(results_df["actual_log10c"], results_df["pred_log10c"]),
        "test/median_fold_error": results_df["fold_error"].median(),
    }

    print()
    print("Overall test-set metrics")
    print(f"R^2 (normalized target): {summary_metrics['test/r2_norm']:.3f}")
    print(f"R^2 (log10c): {summary_metrics['test/r2_log10c']:.3f}")
    print(f"RMSE (log10c): {summary_metrics['test/rmse_log10c']:.3f}")
    print(f"MAE (log10c): {summary_metrics['test/mae_log10c']:.3f}")
    print(f"Median fold error (conc scale): {summary_metrics['test/median_fold_error']:.3f}")

    largest_errors = results_df[
        [
            "species_latin_name",
            "species_group",
            "endpoint",
            "effect",
            "actual_log10c",
            "pred_log10c",
            "abs_error_log10c",
            "fold_error",
            "smiles",
        ]
    ].sort_values("abs_error_log10c", ascending=False).head(LARGEST_ERRORS_N)

    print()
    print(f"Largest {LARGEST_ERRORS_N} test-set errors")
    print(largest_errors.to_string(index=False))

    print()
    print("Group summaries")
    for category in CATEGORICAL_COLS:
        summary = summarize_by_group(results_df, train_df, category, min_count=GROUP_SUMMARY_MIN_COUNT)
        if summary.empty:
            print(f"{category}: no groups with at least {GROUP_SUMMARY_MIN_COUNT} test samples")
            continue
        print(f"{category}:")
        print(summary.head(10).to_string(index=False))

    if SAVE_RESULTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(RESULTS_CSV_PATH, index=False)
        largest_errors.to_csv(LARGEST_ERRORS_CSV_PATH, index=False)
        print()
        print(f"Saved prediction results to {RESULTS_CSV_PATH}")
        print(f"Saved largest errors to {LARGEST_ERRORS_CSV_PATH}")

    return results_df, summary_metrics


def main():
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 80)

    if SET_GLOBAL_SEED:
        set_seed(RANDOM_STATE)

    print("Setup complete")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data file: {DATA_PATH}")

    df_filtered = load_and_filter_data()
    df_processed = preprocess_data(df_filtered)
    df_processed = add_graph_features(df_processed)
    df_processed, df_tax, config_tax, df_categorical, categorical_encoder, config_categorical = build_metadata(
        df_processed
    )

    features = build_graph_features(
        df_processed,
        df_tax,
        TAX_EMBEDDING,
        df_categorical,
        CATEGORICAL_COLS,
        NUMERICAL_COLS,
    )

    print()
    print(f"Graph objects created: {len(features):,}")
    print("Sample graph:")
    print(features[min(16, len(features) - 1)])

    train_dataset, val_dataset, test_dataset = butina_split(
        features,
        stratify_by=STRATIFY_BY,
        frac_train=FRAC_TRAIN,
        frac_test=FRAC_TEST,
        frac_valid=FRAC_VALID,
        cluster_csv_path=CLUSTER_CSV_PATH,
        cluster_col=CLUSTER_COL,
    )
    print_split_info(train_dataset, val_dataset, test_dataset)

    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset)
    model, device, model_info = build_model(features, config_tax, config_categorical)
    model, history = train_model(model, train_loader, val_loader, test_loader, device, model_info, categorical_encoder)
    evaluate_results(model, test_loader, device, df_processed, train_dataset)

    print()
    print("Training finished")
    print(f"Epochs ran: {history['history_all']['epochs_ran']}")
    print(f"Best epoch: {history['history_all']['best_epoch']}")
    print(f"Best monitor value: {history['history_all']['best_monitor_value']}")


if __name__ == "__main__":
    main()
