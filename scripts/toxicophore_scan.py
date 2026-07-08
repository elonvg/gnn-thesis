"""Screen a curated toxicophore (structural-alert) library against the dataset.

Goal: find, per species group, which known toxicophores actually occur in the
molecules the model predicts *well* (so a Shapley test on them is meaningful),
and whether each toxicophore lands inside a single BRICS group (so it is
attributable under the group-Shapley pipeline used in brics-analysis.ipynb).

Alert sources (chosen for this thesis):
  - Kazius et al. 2005 mutagenicity toxicophores (mammalian / rodents)   [hand SMARTS]
  - Aquatic / Verhaar reactive-toxicity alerts (fish, crustaceans, algae)[hand SMARTS]
  - RDKit FilterCatalog BRENK + NIH reactive/toxic alerts                [catalog]

Output: outputs/reports/toxicophore_scan/toxicophore_scan.csv with one row per
(species_group, alert): n_molecules matched among well-predicted rows, and the
fraction of those where the alert sits inside one BRICS group.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------------- #
SPECIES_GROUPS = ["fish", "crustaceans", "algae", "rodents"]
MAX_ABS_ERROR_LOG10C = 0.30      # "well-predicted" threshold, matches the notebook
MIN_MOLECULES = 30               # keep alerts occurring in >= this many molecules
EXPERIMENT_MODEL = "afp-11M-10fold"

# ---------------------------------------------------------------------------- #
# Project root
# ---------------------------------------------------------------------------- #
PROJECT_ROOT = next(
    (c for c in [Path.cwd(), *Path.cwd().parents] if (c / "src").exists()), None
)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate project root (no src/ found).")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.interpretability import get_brics_atom_groups  # noqa: E402

# ---------------------------------------------------------------------------- #
# 1. Curated SMARTS alerts.  Verify against your primary source before citing;
#    tighten/replace with the official Kazius/Toxtree SMARTS files if you have them.
# ---------------------------------------------------------------------------- #
SMARTS_ALERTS = [
    # --- Kazius 2005 mutagenicity toxicophores (mammalian) --------------------
    ("aromatic_nitro",        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][a]",      "DNA-reactive (nitro)",        "Kazius2005"),
    ("aromatic_amine",        "[a][NX3;H1,H2;!$(N-C=O);!$(N-N)]",           "DNA-reactive (arylamine)",    "Kazius2005"),
    ("aromatic_azo",          "[a][NX2]=[NX2][a]",                          "azo / arylamine precursor",   "Kazius2005"),
    ("aliphatic_halide",      "[CX4][Cl,Br,I]",                             "alkylating agent",            "Kazius2005"),
    ("nitrogen_mustard",      "[NX3]([CX4][CX4][Cl,Br,I])",                 "bifunctional alkylator",      "Kazius2005"),
    ("nitroso",               "[#6,#7][NX2]=O",                             "nitroso (DNA-reactive)",      "Kazius2005"),
    ("azide",                 "[NX2,NX1-]=[NX2+]=[NX1-]",                   "azide",                       "Kazius2005"),
    ("aromatic_hydroxylamine","[a][NX3;H1][OX2;H1]",                        "N-hydroxy arylamine",         "Kazius2005"),
    # --- Three-membered strained electrophiles (both mammalian & aquatic) -----
    ("epoxide",               "[OX2r3]1[#6r3][#6r3]1",                      "strained electrophile",       "reactive"),
    ("aziridine",             "[NX3r3]1[#6r3][#6r3]1",                      "strained electrophile",       "reactive"),
    # --- Aquatic / Verhaar reactive-toxicity alerts ---------------------------
    ("michael_acceptor",      "[CX3]=[CX3][CX3]=[OX1]",                     "Michael acceptor (enone)",    "aquatic"),
    ("acrylonitrile",         "[CX3]=[CX3]C#N",                             "Michael acceptor (nitrile)",  "aquatic"),
    ("aldehyde",              "[CX3H1](=O)[#6]",                            "Schiff-base / electrophile",  "aquatic"),
    ("isothiocyanate",        "[NX2]=[CX2]=[SX1]",                          "electrophile (SCN)",          "aquatic"),
    ("isocyanate",            "[NX2]=[CX2]=[OX1]",                          "electrophile (NCO)",          "aquatic"),
    ("organophosphate",       "[PX4](=[OX1,SX1])([OX2,SX2])[OX2,SX2]",      "AChE inhibitor (OP)",         "aquatic"),
    ("carbamate",             "[NX3][CX3](=[OX1])[OX2][#6]",                "AChE inhibitor (carbamate)",  "aquatic"),
    ("quinone",               "O=[#6]1[#6]=[#6][#6](=O)[#6]=[#6]1",         "redox-cycling quinone",       "aquatic"),
    ("phenol",                "[OX2H][c]",                                  "polar narcotic / uncoupler",  "aquatic"),
]
COMPILED = [
    (name, Chem.MolFromSmarts(sm), mech, src)
    for name, sm, mech, src in SMARTS_ALERTS
]
for name, patt, _, sm in [(n, p, m, s) for n, p, m, s in COMPILED]:
    if patt is None:
        raise ValueError(f"SMARTS failed to compile for alert {name!r}")

# ---------------------------------------------------------------------------- #
# 2. RDKit FilterCatalog: BRENK + NIH
# ---------------------------------------------------------------------------- #
_params = FilterCatalogParams()
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
CATALOG = FilterCatalog.FilterCatalog(_params)


def _slug(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower()).strip("_")


def alert_matches(mol):
    """Yield (alert_name, mechanism, source, [frozenset(atom_idx), ...])."""
    for name, patt, mech, src in COMPILED:
        hits = mol.GetSubstructMatches(patt, uniquify=True)
        if hits:
            yield name, mech, src, [frozenset(h) for h in hits]

    for entry in CATALOG.GetMatches(mol):
        name = f"{_slug(entry.GetDescription())}"
        atom_sets = []
        for fm in entry.GetFilterMatches(mol):
            atom_sets.append(frozenset(p[1] for p in fm.atomPairs))
        yield f"catalog__{name}", "reactive/tox (catalog)", "BRENK/NIH", atom_sets


# ---------------------------------------------------------------------------- #
# 3. BRICS containment
# ---------------------------------------------------------------------------- #
_brics_cache: dict[str, list[frozenset]] = {}


def brics_groups(smiles: str, mol) -> list[frozenset]:
    if smiles not in _brics_cache:
        _brics_cache[smiles] = [frozenset(g) for g in get_brics_atom_groups(mol)]
    return _brics_cache[smiles]


def within_single_group(atom_sets, groups) -> bool:
    """True if any match is fully contained in one BRICS group."""
    for match in atom_sets:
        if any(match <= g for g in groups):
            return True
    return False


# ---------------------------------------------------------------------------- #
# 4. Load well-predicted molecules per species group
# ---------------------------------------------------------------------------- #
def load_predictions() -> pd.DataFrame:
    model_dir = PROJECT_ROOT / "outputs" / "experiments" / EXPERIMENT_MODEL
    frames = [pd.read_csv(p, compression="gzip") for p in sorted(model_dir.glob("fold_*_val_predictions.csv.gz"))]
    if not frames:
        raise FileNotFoundError(f"No prediction files in {model_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["species_group"] = df["species_group"].astype("string").str.strip().str.lower()
    df["abs_error_log10c"] = pd.to_numeric(df["abs_error_log10c"], errors="coerce")
    return df


def main() -> None:
    preds = load_predictions()
    records = []

    for group in SPECIES_GROUPS:
        sub = preds[
            (preds["species_group"] == group)
            & (preds["abs_error_log10c"] <= MAX_ABS_ERROR_LOG10C)
        ]
        smiles_list = sub["SMILES"].dropna().unique().tolist()
        print(f"{group:12s}: {len(smiles_list):5d} distinct well-predicted molecules")

        counts: dict[str, int] = {}
        single: dict[str, int] = {}
        meta: dict[str, tuple] = {}

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            groups = None
            for name, mech, src, atom_sets in alert_matches(mol):
                counts[name] = counts.get(name, 0) + 1
                meta[name] = (mech, src)
                if groups is None:
                    groups = brics_groups(smiles, mol)
                if within_single_group(atom_sets, groups):
                    single[name] = single.get(name, 0) + 1

        for name, n in counts.items():
            n_single = single.get(name, 0)
            mech, src = meta[name]
            records.append({
                "species_group": group,
                "alert": name,
                "mechanism": mech,
                "source": src,
                "n_molecules": n,
                "single_brics_group_frac": round(n_single / n, 3),
            })

    out = pd.DataFrame(records)
    out = out[out["n_molecules"] >= MIN_MOLECULES]
    out = out.sort_values(["species_group", "n_molecules"], ascending=[True, False])

    out_dir = PROJECT_ROOT / "outputs" / "reports" / "toxicophore_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "toxicophore_scan.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} (group, alert) rows -> {out_path}\n")

    for group in SPECIES_GROUPS:
        g = out[out["species_group"] == group].head(15)
        if g.empty:
            continue
        print(f"=== {group} — top alerts by occurrence ===")
        print(g[["alert", "mechanism", "n_molecules", "single_brics_group_frac"]].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
