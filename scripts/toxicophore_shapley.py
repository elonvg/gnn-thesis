#!/usr/bin/env python3
"""Two-player Shapley toxicophore worker.

For one (species_group, endpoint, conc_unit) slice, this:
  1. rebuilds the same data / graphs / model init as brics-analysis.ipynb,
  2. flags toxicophores with BRENK + NIH (RDKit) + ToxAlerts (filtered endpoints),
  3. samples N well-predicted, alert-bearing molecules PER FOLD,
  4. verifies atom-index alignment (graph node i == RDKit atom i) and fails loudly,
  5. computes a TWO-PLAYER Shapley for each toxicophore occurrence:
         phi = 0.5 * ((tox_alone - empty) + (full - rest))
         tox_shapley = -phi          (positive => the fragment makes it MORE toxic)
  6. saves CSVs to outputs/reports/brics_fragment_analysis/<species_group>/ (overwrites).

This is a WORKER: it does the heavy compute and writes CSVs. Plot from the CSVs
elsewhere. Set the slice / sample size with the CONFIG block or CLI flags, e.g.

    python scripts/toxicophore_shapley.py --species-group fish \
        --endpoint EC50 --conc-unit mg/L --n-per-fold 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import (
    FilterCatalogEntry,
    FilterCatalogParams,
    SmartsMatcher,
)

RDLogger.DisableLog("rdApp.*")

# ============================== CONFIG (defaults) ============================== #
SPECIES_GROUP = "rodents"
ENDPOINT = "EC50"
CONC_UNIT = "mg/kg"
N_PER_FOLD = 5                 # molecules sampled PER FOLD (small for a test run)
FOLDS = None                   # None = all folds in afp_df; or e.g. [0, 1]
MAX_ABS_ERROR = 0.75           # keep only molecules with |pred - actual| <= this
RANDOM_STATE = 11

# Bottom-up (BRICS group-Shapley) settings
GROUP_SHAPLEY_N_SAMPLES = 64   # permutations for the BRICS group-Shapley estimate
MIN_BRICS_GROUPS = 2           # molecules with fewer BRICS groups skip the bottom-up test
MAX_BRICS_GROUPS = 12          # ... and molecules with more (keeps compute bounded)
MAX_MOLECULE_ATOMS = 80        # skip bottom-up for very large molecules

MODEL_NAME = "afp-11M-10fold"
CHECKPOINT_TEMPLATE = "AttentiveFP-11M-fold{fold}.pt"

# ToxAlerts: the four downloaded files are identical, so we load one and filter by
# the PROPERTY column to the endpoints we chose.
TOXALERTS_FILE = "Data/ToxAlerts/ToxAlerts_Acute_Aquatic_Toxicity.csv"
TOXALERTS_ENDPOINTS = [
    "Acute Aquatic Toxicity",
    "Reactive, unstable, toxic",
    "Potential electrophilic agents",
    "Skin sensitization",
]

# Dataset-level filters that define the model's data universe. MUST match the
# notebook so row_id indexes line up with the trained model's predictions.
DATA_FILTERS = {
    "duration_unit": ["h"],
    "effect": ["MOR", "POP", "GRO", "BEH", "REP", "ITX", "PHY", "DVP", "MPH"],
}
N_SAMPLES = None
SPLIT_SALTS = False
REMOVE_LONE = False
REMOVE_METALS = False
MAX_CONC_VALUE = 10000
DURATION_FILL_VALUE = 1e-6
MAX_DURATION_HOURS = 9000.0

# Atom feature layout (must match the trained model).
ATOM_FEATURES = (
    "atomic_num", "degree", "formal_charge", "num_hs", "hybridization",
    "is_aromatic", "is_in_ring", "atomic_mass", "period", "group",
    "covalent_radius", "vdw_radius", "is_metal",
)
BOND_FEATURES = ("bond_order", "is_conjugated", "is_in_ring", "stereo")
MOLECULE_CATEGORICAL_COLS = ["is_salt", "has_metal", "is_single_node"]
MOLECULE_NUMERICAL_COLS = ["fragment_count", "log10_mol_weight", "formal_charge"]
EXP_CATEGORICAL_COLS = ["species_group", "conc_unit", "endpoint", "effect"]

# atomic_num is the FIRST atom feature: one-hot over elements 1..118 -> 118 columns.
ATOMIC_NUM_WIDTH = 118
# ============================================================================== #


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src").exists() and (candidate / "Data").exists():
            return candidate
    raise RuntimeError("Could not locate the project root (needs src/ and Data/).")


# ------------------------------- alert catalogs ------------------------------- #
def build_alert_catalogs(project_root: Path):
    """Return [(source, FilterCatalog), ...] plus the frozen ToxAlerts table."""
    # BRENK + NIH (built into RDKit)
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
    brenk_nih = FilterCatalog.FilterCatalog(params)

    # ToxAlerts filtered to our endpoints
    tox = pd.read_csv(project_root / TOXALERTS_FILE)
    tox = (
        tox[tox["PROPERTY"].isin(TOXALERTS_ENDPOINTS)]
        .dropna(subset=["SMARTS"])
        .drop_duplicates("SMARTS")
        .reset_index(drop=True)
    )
    tox_catalog = FilterCatalog.FilterCatalog()
    n_bad = 0
    kept = []
    for _, row in tox.iterrows():
        smarts = str(row["SMARTS"])
        if Chem.MolFromSmarts(smarts) is None:
            n_bad += 1
            continue
        name = f'{row["Alert ID"]} | {row["PROPERTY"]} | {row["NAME"]}'
        tox_catalog.AddEntry(FilterCatalogEntry(name, SmartsMatcher(name, smarts)))
        kept.append(row)

    brenk_nih_names = [
        brenk_nih.GetEntryWithIdx(i).GetDescription()
        for i in range(brenk_nih.GetNumEntries())
    ]
    print(
        f"Alerts: BRENK+NIH={brenk_nih.GetNumEntries()}, "
        f"ToxAlerts kept={tox_catalog.GetNumEntries()} (skipped {n_bad} unparseable)"
    )
    frozen = pd.DataFrame(kept) if kept else pd.DataFrame(columns=tox.columns)
    return [("BRENK/NIH", brenk_nih), ("ToxAlerts", tox_catalog)], frozen, brenk_nih_names, n_bad


def toxicophore_hits(mol, catalogs):
    """Every alert match with atoms. Returns list of dicts {source, property, alert, atoms}."""
    hits = []
    for source, cat in catalogs:
        for entry in cat.GetMatches(mol):
            desc = entry.GetDescription()
            if source == "ToxAlerts":
                parts = [p.strip() for p in desc.split("|")]
                prop = parts[1] if len(parts) >= 2 else "ToxAlerts"
                name = parts[2] if len(parts) >= 3 else desc
            else:
                prop, name = "BRENK/NIH", desc
            for m in entry.GetFilterMatches(mol):
                atoms = tuple(sorted({j for _, j in m.atomPairs}))
                if atoms:
                    hits.append({"source": source, "property": prop, "alert": name, "atoms": atoms})
    return hits


def unique_atom_sets(hits):
    """Collapse alerts that hit the SAME atoms. Returns {atoms: set((source, property, alert))}."""
    merged: dict[tuple, set] = {}
    for h in hits:
        merged.setdefault(h["atoms"], set()).add((h["source"], h["property"], h["alert"]))
    return merged


def has_toxicophore(smiles, catalogs) -> bool:
    mol = Chem.MolFromSmiles(str(smiles))
    return mol is not None and len(toxicophore_hits(mol, catalogs)) > 0


# ------------------------------ atom alignment ------------------------------- #
def graph_atomic_numbers(graph):
    return (graph.x[:, :ATOMIC_NUM_WIDTH].argmax(dim=1) + 1).tolist()


def assert_alignment(row_id, smiles, graphs):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise AssertionError(f"row {row_id}: unparseable SMILES {smiles!r}")
    g = graphs[row_id]
    if g.x.shape[0] != mol.GetNumAtoms():
        raise AssertionError(
            f"row {row_id}: graph has {g.x.shape[0]} nodes but SMILES has "
            f"{mol.GetNumAtoms()} atoms ({smiles!r})"
        )
    z_graph = graph_atomic_numbers(g)
    z_mol = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if z_graph != z_mol:
        raise AssertionError(
            f"row {row_id}: atom order/elements differ ({smiles!r})\n"
            f"  graph={z_graph}\n  mol  ={z_mol}"
        )


# ------------------------------- data / model -------------------------------- #
def init_data(project_root):
    """Rebuild df_processed, graphs, encoders (mirrors brics-analysis.ipynb)."""
    from src.data.io import load_data
    from src.data.cleaning import process_data
    from src.data.features_graph import GraphFeaturizer
    from src.data.features_mol import add_molecule_metadata
    from src.data.metadata import sequential_encoder, build_config
    from src.data.graph_building import build_graphs

    df_all = load_data(project_root / "Data" / "toxicity_all.csv")
    df_processed = process_data(
        df_all, n_samples=N_SAMPLES, random_state=RANDOM_STATE, filters=DATA_FILTERS,
        require_duration=False, require_taxid=True, split_salts=SPLIT_SALTS,
        remove_lone=REMOVE_LONE, remove_metals=REMOVE_METALS, max_conc_value=MAX_CONC_VALUE,
        duration_fill_value=DURATION_FILL_VALUE, max_duration_hours=MAX_DURATION_HOURS,
        log_transform_duration=True, keep_duration_raw=True,
    )

    featurizer = GraphFeaturizer(ATOM_FEATURES, BOND_FEATURES)
    df_processed["features"] = df_processed["SMILES"].apply(featurizer.featurize)
    df_processed = add_molecule_metadata(
        df_processed, categorical_cols=MOLECULE_CATEGORICAL_COLS,
        numerical_cols=MOLECULE_NUMERICAL_COLS,
    )

    categorical_cols = EXP_CATEGORICAL_COLS + MOLECULE_CATEGORICAL_COLS
    df_categorical = df_processed[categorical_cols].copy()
    df_categorical, _ = sequential_encoder(df_categorical, categorical_cols)
    config_categorical = build_config(df_categorical, categorical_cols)
    numerical_cols = ["duration"] + MOLECULE_NUMERICAL_COLS

    graphs = build_graphs(df_processed, df_categorical, categorical_cols, numerical_cols)
    print(f"Built {len(graphs):,} graphs.")
    return df_processed, graphs, config_categorical, numerical_cols


def make_build_model(config_categorical, numerical_cols, graphs, project_root, device):
    from src.models.attentive_fp import AttentiveFP
    from src.models.toxicity_model import ToxicityModel
    from src.models.meta_encoder import MetaEncoder

    pretrained_taxid_path = project_root / "Data" / "moredata" / "pretrained_tax_emb.pkl.zip"
    atom_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]

    def build_model():
        meta = MetaEncoder(
            pretrained_taxid_path=pretrained_taxid_path, pretrained_tax_dim=768,
            pretrained_taxid_output_dim=512, config_categorical=config_categorical,
            categorical_output_dim=128, numerical_columns=numerical_cols,
            numeric_output_dim=128, dropout=0.3,
        ).to(device)
        gnn = AttentiveFP(
            in_channels=atom_dim, edge_dim=edge_dim, hidden_channels=512,
            out_channels=512, num_layers=3, num_timesteps=2, dropout=0.3,
        ).to(device)
        return ToxicityModel(gnn, meta, hidden_dim=1024).to(device)

    return build_model


def load_model_folds(project_root, model_name):
    exp_dir = project_root / "outputs" / "experiments" / model_name
    pat = re.compile(r"fold_(\d+)_val_predictions\.csv\.gz$")
    frames = []
    for path in sorted(exp_dir.glob("fold_*_val_predictions.csv.gz")):
        fold = int(pat.search(path.name).group(1))
        df = pd.read_csv(path, compression="gzip")
        df["fold"] = fold
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No fold prediction files in {exp_dir}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------- sampling ---------------------------------- #
def select_molecules(afp_df, species_group, endpoint, conc_unit,
                     max_abs_error, n_per_fold, folds, seed):
    work = afp_df.copy()
    work["row_id"] = pd.to_numeric(work["row_id"], errors="coerce").astype("Int64")
    work["fold"] = pd.to_numeric(work["fold"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["row_id", "fold"]).copy()
    work["row_id"] = work["row_id"].astype(int)
    work["fold"] = work["fold"].astype(int)

    target = "actual_log10c" if "actual_log10c" in work.columns else "log10c"
    work["actual_log10c"] = pd.to_numeric(work[target], errors="coerce")
    work["pred_log10c"] = pd.to_numeric(work["pred_log10c"], errors="coerce")
    if "abs_error_log10c" in work.columns:
        work["abs_error_log10c"] = pd.to_numeric(work["abs_error_log10c"], errors="coerce")
    else:
        work["abs_error_log10c"] = np.nan
    miss = work["abs_error_log10c"].isna()
    work.loc[miss, "abs_error_log10c"] = (
        work.loc[miss, "pred_log10c"] - work.loc[miss, "actual_log10c"]
    ).abs()

    # case-insensitive slice filters
    for col, val in [("species_group", species_group), ("endpoint", endpoint), ("conc_unit", conc_unit)]:
        work = work[work[col].astype("string").str.strip().str.lower() == str(val).strip().lower()]

    work = work.dropna(subset=["actual_log10c", "pred_log10c", "abs_error_log10c"])
    work = work[work["abs_error_log10c"] <= max_abs_error]
    if folds is not None:
        work = work[work["fold"].isin([int(f) for f in folds])]
    n_after_filter = len(work)

    # unique molecules, keep the best-predicted row per SMILES
    work = work.sort_values(["SMILES", "abs_error_log10c"]).drop_duplicates("SMILES", keep="first")
    n_unique = len(work)
    if work.empty:
        raise ValueError(
            "No well-predicted molecules for this slice. Loosen MAX_ABS_ERROR "
            "or check species_group/endpoint/conc_unit."
        )

    # Representative sample (NOT pre-filtered to toxicophore-bearing): bottom-up
    # BRICS discovery must see a representative set; top-down runs on whichever of
    # these contain a toxicophore.
    sampled = (
        work.groupby("fold", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), n_per_fold), random_state=seed))
        .reset_index(drop=True)
    )
    stats = {
        "rows_after_slice_and_accuracy_filter": int(n_after_filter),
        "unique_molecules": int(n_unique),
        "sampled_molecules": int(len(sampled)),
        "folds_present": sorted(int(f) for f in sampled["fold"].unique()),
    }
    return sampled, stats


def brics_fragment_smiles(mol, atom_indices):
    return Chem.MolFragmentToSmiles(
        mol, atomsToUse=sorted(int(a) for a in atom_indices),
        canonical=True, isomericSmiles=True,
    )


# ------------------------------ run both analyses ---------------------------- #
def run_analyses(sampled, graphs, catalogs, build_model, project_root, device):
    """Per sampled molecule, run TOP-DOWN (two-player toxicophore Shapley) and
    BOTTOM-UP (BRICS group-Shapley), reusing one model per fold."""
    from src.visualization.interpretability import (
        predict_with_atom_mask,
        group_shapley_permutation,
        get_brics_atom_groups,
    )

    model_dir = project_root / "outputs" / "models" / MODEL_NAME
    tox_contribs, tox_alerts_long, brics_contribs = [], [], []
    occ_id = 0
    n_alert_bearing = n_brics_done = n_brics_skipped = 0

    for fold, fold_rows in sampled.groupby("fold", sort=True):
        model = build_model()
        ckpt_path = model_dir / CHECKPOINT_TEMPLATE.format(fold=int(fold))
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        print(f"Fold {fold}: loaded {ckpt_path.name} for {len(fold_rows)} molecules")

        for i, row in fold_rows.reset_index(drop=True).iterrows():
            row_id = int(row["row_id"])
            smiles = row["SMILES"]
            mol = Chem.MolFromSmiles(str(smiles))
            graph = graphs[row_id].clone().to(device)
            n_atoms = graph.x.size(0)
            baseline = torch.zeros_like(graph.x)

            meta = {
                "row_id": row_id, "fold": int(fold), "SMILES": smiles,
                "chemical_name": row.get("chemical_name", np.nan),
                "CAS": row.get("CAS", np.nan),
                "species_group": row.get("species_group", np.nan),
                "endpoint": row.get("endpoint", np.nan),
                "conc_unit": row.get("conc_unit", np.nan),
                "effect": row.get("effect", np.nan),
                "actual_log10c": row["actual_log10c"],
                "saved_pred_log10c": row["pred_log10c"],
                "abs_error_log10c": row["abs_error_log10c"],
                "n_atoms": n_atoms,
            }

            # ---------- TOP-DOWN: two-player toxicophore Shapley ----------
            atom_sets = unique_atom_sets(toxicophore_hits(mol, catalogs))
            if atom_sets:
                n_alert_bearing += 1
                all_true = torch.ones(n_atoms, dtype=torch.bool, device=device)
                all_false = torch.zeros(n_atoms, dtype=torch.bool, device=device)
                with torch.no_grad():
                    pred_empty = predict_with_atom_mask(model, graph, all_false, baseline)
                    pred_full = predict_with_atom_mask(model, graph, all_true, baseline)

                for atoms, labels in atom_sets.items():
                    tox_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
                    tox_mask[list(atoms)] = True
                    with torch.no_grad():
                        pred_tox_alone = predict_with_atom_mask(model, graph, tox_mask, baseline)
                        pred_rest = predict_with_atom_mask(model, graph, ~tox_mask, baseline)

                    phi = 0.5 * ((pred_tox_alone - pred_empty) + (pred_full - pred_rest))
                    tox = -phi
                    sources = sorted({s for s, _, _ in labels})
                    properties = sorted({p for _, p, _ in labels})
                    alert_names = sorted({a for _, _, a in labels})

                    tox_contribs.append({
                        **meta, "occ_id": occ_id,
                        "atom_indices": ",".join(map(str, atoms)),
                        "n_tox_atoms": len(atoms), "n_alerts": len(alert_names),
                        "sources": "|".join(sources), "properties": "|".join(properties),
                        "alerts": "|".join(alert_names),
                        "pred_empty_log10c": pred_empty, "pred_full_log10c": pred_full,
                        "pred_tox_alone_log10c": pred_tox_alone, "pred_rest_log10c": pred_rest,
                        "phi_tox_log10c": phi, "tox_shapley_log10c": tox,
                        "tox_shapley_per_atom_log10c": tox / max(len(atoms), 1),
                        "pred_full_minus_saved": pred_full - row["pred_log10c"],
                    })
                    for source, prop, alert in labels:
                        tox_alerts_long.append({
                            "occ_id": occ_id, "row_id": row_id, "fold": int(fold),
                            "SMILES": smiles, "species_group": row.get("species_group", np.nan),
                            "source": source, "property": prop, "alert": alert,
                            "n_tox_atoms": len(atoms), "tox_shapley_log10c": tox,
                            "tox_shapley_per_atom_log10c": tox / max(len(atoms), 1),
                        })
                    occ_id += 1

            # ---------- BOTTOM-UP: BRICS group-Shapley ----------
            groups = get_brics_atom_groups(mol)
            if (len(groups) < MIN_BRICS_GROUPS or len(groups) > MAX_BRICS_GROUPS
                    or n_atoms > MAX_MOLECULE_ATOMS):
                n_brics_skipped += 1
            else:
                shapley = group_shapley_permutation(
                    model, graph, groups, n_samples=GROUP_SHAPLEY_N_SAMPLES
                ).detach().cpu().numpy()
                tox_vals = -shapley
                top_idx = int(np.argmax(tox_vals))
                for gi, (atoms, shap, tval) in enumerate(zip(groups, shapley, tox_vals)):
                    brics_contribs.append({
                        **meta, "group_idx": gi, "n_brics_groups": len(groups),
                        "fragment_smiles": brics_fragment_smiles(mol, atoms),
                        "atom_indices": ",".join(map(str, sorted(atoms))),
                        "n_group_atoms": len(atoms),
                        "group_shapley_log10c": float(shap),
                        "tox_shapley_log10c": float(tval),
                        "tox_shapley_per_atom_log10c": float(tval) / max(len(atoms), 1),
                        "is_top_toxic_group": gi == top_idx,
                    })
                n_brics_done += 1

            print(f"  {i + 1:>3}/{len(fold_rows)} row_id={row_id} "
                  f"tox_sets={len(atom_sets)} brics_groups={len(groups)} atoms={n_atoms}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_stats = {
        "sampled_molecules": int(len(sampled)),
        "alert_bearing_sampled": int(n_alert_bearing),
        "brics_analyzed": int(n_brics_done),
        "brics_skipped_guards": int(n_brics_skipped),
        "toxicophore_occurrences": int(len(tox_contribs)),
        "brics_fragment_rows": int(len(brics_contribs)),
    }
    return (pd.DataFrame(tox_contribs), pd.DataFrame(tox_alerts_long),
            pd.DataFrame(brics_contribs), run_stats)


# ----------------------------------- main ------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--species-group", default=SPECIES_GROUP)
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--conc-unit", default=CONC_UNIT)
    p.add_argument("--n-per-fold", type=int, default=N_PER_FOLD)
    p.add_argument("--folds", type=int, nargs="*", default=FOLDS,
                   help="folds to use (default: all present)")
    p.add_argument("--max-abs-error", type=float, default=MAX_ABS_ERROR)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return p.parse_args()


def main():
    args = parse_args()
    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Project root: {project_root}\nDevice: {device}")
    print(f"Slice: species_group={args.species_group!r} endpoint={args.endpoint!r} "
          f"conc_unit={args.conc_unit!r} | n_per_fold={args.n_per_fold} "
          f"max_abs_error={args.max_abs_error}")

    catalogs, tox_frozen, brenk_nih_names, n_bad_smarts = build_alert_catalogs(project_root)
    df_processed, graphs, config_categorical, numerical_cols = init_data(project_root)
    afp_df = load_model_folds(project_root, MODEL_NAME)
    build_model = make_build_model(config_categorical, numerical_cols, graphs, project_root, device)

    sampled, stats = select_molecules(
        afp_df, args.species_group, args.endpoint, args.conc_unit,
        args.max_abs_error, args.n_per_fold, args.folds, args.random_state,
    )
    print(f"Sampling: {stats}")

    # Hard atom-alignment gate BEFORE any Shapley compute.
    for _, row in sampled.iterrows():
        assert_alignment(int(row["row_id"]), row["SMILES"], graphs)
    print(f"Atom alignment verified for all {len(sampled)} sampled molecules.")

    tox_contribs, tox_alerts_long, brics_contribs, run_stats = run_analyses(
        sampled, graphs, catalogs, build_model, project_root, device,
    )

    # ------------------------------- save ------------------------------------ #
    out_dir = project_root / "outputs" / "reports" / "brics_fragment_analysis" / args.species_group
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top-down (toxicophore two-player Shapley)
    tox_contribs.to_csv(out_dir / "toxicophore_contributions.csv", index=False)
    tox_alerts_long.to_csv(out_dir / "toxicophore_alerts_long.csv", index=False)
    # Bottom-up (BRICS group-Shapley)
    brics_contribs.to_csv(out_dir / "brics_contributions.csv", index=False)
    # Frozen alert set (for the methods section)
    tox_frozen.to_csv(out_dir / "alert_set_toxalerts.csv", index=False)
    pd.DataFrame({"source": "BRENK/NIH", "alert": brenk_nih_names}).to_csv(
        out_dir / "alert_set_brenk_nih.csv", index=False
    )

    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "methods": {
            "top_down": "two_player_shapley (tox atoms vs rest); tox = -phi",
            "bottom_up": f"BRICS group_shapley_permutation (n_samples={GROUP_SHAPLEY_N_SAMPLES}); tox = -shapley",
        },
        "slice": {
            "species_group": args.species_group,
            "endpoint": args.endpoint,
            "conc_unit": args.conc_unit,
        },
        "sampling": {
            "n_per_fold": args.n_per_fold,
            "folds": args.folds,
            "max_abs_error": args.max_abs_error,
            "random_state": args.random_state,
            **stats,
            **run_stats,
        },
        "bottom_up_guards": {
            "min_brics_groups": MIN_BRICS_GROUPS,
            "max_brics_groups": MAX_BRICS_GROUPS,
            "max_molecule_atoms": MAX_MOLECULE_ATOMS,
        },
        "alerts": {
            "brenk_nih": len(brenk_nih_names),
            "toxalerts_kept": int(len(tox_frozen)),
            "toxalerts_endpoints": TOXALERTS_ENDPOINTS,
            "toxalerts_skipped_unparseable": int(n_bad_smarts),
        },
        "model": MODEL_NAME,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\n================= DONE =================")
    print(f"Sampled molecules: {len(sampled)}  "
          f"(alert-bearing: {run_stats['alert_bearing_sampled']}, "
          f"BRICS analyzed: {run_stats['brics_analyzed']}, "
          f"skipped by guards: {run_stats['brics_skipped_guards']})")
    print(f"Top-down toxicophore occurrences: {len(tox_contribs):,}")
    if not tox_contribs.empty:
        print(f"  tox_shapley_log10c: mean={tox_contribs['tox_shapley_log10c'].mean():+.4f} "
              f"median={tox_contribs['tox_shapley_log10c'].median():+.4f}")
        chk = tox_contribs["pred_full_minus_saved"].abs().max()
        print(f"  max |pred_full - saved_pred| (sanity, expect ~0): {chk:.4f}")
    print(f"Bottom-up BRICS fragment rows: {len(brics_contribs):,}")
    if not brics_contribs.empty:
        print(f"  tox_shapley_log10c: mean={brics_contribs['tox_shapley_log10c'].mean():+.4f} "
              f"median={brics_contribs['tox_shapley_log10c'].median():+.4f}")
    print(f"Saved to: {out_dir}")
    for name in ["toxicophore_contributions.csv", "toxicophore_alerts_long.csv",
                 "brics_contributions.csv", "alert_set_toxalerts.csv",
                 "alert_set_brenk_nih.csv", "metadata.json"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
