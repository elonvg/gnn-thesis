import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image
from matplotlib import cm, colors
import math
from rdkit import Chem
import pandas as pd

import torch

def draw_atom_heatmap_gaussian(
    smiles,
    explanation,
    reduce="sum_abs",
    cmap_name="plasma",
    size=(700, 500),
    sigma_frac=0.07,       # Gaussian radius as fraction of image width
    alpha=0.65,            # heatmap overlay opacity
    use_percentile=True,
    bg_color=(1, 1, 1),    # white molecule background
):
    composite, atom_scores, vmin, vmax = _make_composite(smiles, explanation, reduce, cmap_name, size, sigma_frac, alpha, use_percentile)

    # if use_percentile:
    #     vmin, vmax = np.percentile(atom_scores, [5, 95])
    # else:
    #     vmin, vmax = atom_scores.min(), atom_scores.max()

    # if vmin == vmax:
    #     vmin = vmax - 1e-6

    # --- Plot ---
    fig, axes = plt.subplots(
        2, 1,
        figsize=(size[0] / 100, (size[1] + 60) / 100),
        gridspec_kw={"height_ratios": [size[1], 60], "hspace": 0.03},
        dpi=100,
    )

    ax_img, ax_cb = axes

    ax_img.imshow(composite, origin="upper", interpolation="bilinear")
    ax_img.axis("off")
    ax_img.set_title(f"{smiles}", fontsize=13, pad=10)

    # Colorbar
    sm = cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        cmap=cm.get_cmap(cmap_name),
    )
    cb = plt.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_label("Atom importance", fontsize=10, labelpad=4)
    cb.ax.tick_params(labelsize=9)

    plt.savefig("atom_heatmap.png", bbox_inches="tight", dpi=150)
    plt.show()
    return atom_scores


def _make_composite(smiles, explanation, reduce, cmap_name, size, sigma_frac, alpha, use_percentile):
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)

    node_mask = explanation.node_mask.detach().cpu()
    if reduce == "sum_abs":
        atom_scores = node_mask.abs().sum(dim=1).numpy()
    elif reduce == "mean_abs":
        atom_scores = node_mask.abs().mean(dim=1).numpy()
    elif reduce == "max_abs":
        atom_scores = node_mask.abs().max(dim=1).values.numpy()
    else:
        raise ValueError(reduce)

    if use_percentile:
        vmin, vmax = np.percentile(atom_scores, [5, 95])
    else:
        vmin, vmax = atom_scores.min(), atom_scores.max()

    if vmin == vmax:
        vmin = vmax - 1e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    # --- Render clean molecule SVG → raster via RDKit ---
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.padding = 0.1
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg_bytes = drawer.GetDrawingText().encode()

    # Convert SVG → PIL image (requires cairosvg or svglib)
    try:
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_bytes, output_width=size[0], output_height=size[1])
        mol_img = np.array(Image.open(BytesIO(png_data)).convert("RGBA")).astype(float) / 255.0
    except ImportError:
        # fallback: use RDKit PNG renderer
        drawer2 = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer2.drawOptions().useBWAtomPalette()
        drawer2.DrawMolecule(mol)
        drawer2.FinishDrawing()
        mol_img = np.array(Image.open(BytesIO(drawer2.GetDrawingText())).convert("RGBA")).astype(float) / 255.0

    H, W = size[1], size[0]

    # --- Get 2-D atom positions in pixel space ---
    conf = mol.GetConformer()
    drawer_tmp = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer_tmp.drawOptions().useBWAtomPalette()
    drawer_tmp.DrawMolecule(mol)
    drawer_tmp.FinishDrawing()

    atom_positions = []
    for i in range(mol.GetNumAtoms()):
        pt = drawer_tmp.GetDrawCoords(i)
        atom_positions.append((pt.x, pt.y))

    # --- Build Gaussian heatmap ---
    sigma = sigma_frac * W
    xs = np.linspace(0, W - 1, W)
    ys = np.linspace(0, H - 1, H)
    xx, yy = np.meshgrid(xs, ys)
    heat = np.zeros((H, W), dtype=float)

    for i, (px, py) in enumerate(atom_positions):
        w = float(norm(atom_scores[i]))
        heat += w * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2))

    # Normalise to [0, 1]
    if heat.max() > 0:
        heat /= heat.max()

    # --- Map heat → RGBA using colormap ---
    cmap = cm.get_cmap(cmap_name)

    # Custom colormap: fully transparent at 0, opaque at 1
    colors_list = cmap(np.linspace(0, 1, 256))
    colors_list[:, 3] = np.linspace(0, alpha, 256)   # ramp alpha with intensity
    custom_cmap = LinearSegmentedColormap.from_list("heat_alpha", colors_list)

    heat_rgba = custom_cmap(heat)   # shape (H, W, 4)

    # --- Composite heatmap over molecule image ---
    # Standard alpha blending: out = heat_fg + mol_bg * (1 - heat_alpha)
    fg_alpha = heat_rgba[..., 3:4]
    fg_rgb   = heat_rgba[..., :3]
    bg_rgb   = mol_img[..., :3]

    composite = fg_rgb * fg_alpha + bg_rgb * (1 - fg_alpha)
    return composite, atom_scores, vmin, vmax

def draw_atom_heatmap_grid(
    graph_ids,
    dataset,
    explainer,
    reduce="sum_abs",
    cmap_name="plasma",
    size=(700, 500),
    sigma_frac=0.06,
    alpha=0.70,
    use_percentile=True,
    ncols=3,
    title=None,
):
    nrows = math.ceil(len(graph_ids) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * size[0] / 100, nrows * size[1] / 100),
        dpi=100,
    )
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    for ax in axes:
        ax.axis("off")

    for plot_idx, graph_id in enumerate(graph_ids):
        graph = dataset[graph_id]

        if not hasattr(graph, "batch") or graph.batch is None:
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
        
        explanation = explainer(
            graph.x,
            graph.edge_index,
            edge_attr=graph.edge_attr,
            virtual_edge_index=graph.virtual_edge_index,
            virtual_edge_attr=graph.virtual_edge_attr,
            fragment_id=graph.fragment_id,
            batch=graph.batch,
        )

        composite, atom_scores, vmin, vmax = _make_composite(
            graph.smiles, explanation, reduce, cmap_name, size, sigma_frac, alpha, use_percentile
        )

        axes[plot_idx].imshow(composite, origin="upper", interpolation="bilinear")
        axes[plot_idx].set_title(f"{graph.smiles}", fontsize=9, pad=4)
        axes[plot_idx].axis("off")

    # shared colorbar along the bottom
    node_masks = []
    for graph_id in graph_ids:
        graph = dataset[graph_id]

        explanation = explainer(
            graph.x,
            graph.edge_index,
            edge_attr=graph.edge_attr,
            virtual_edge_index=graph.virtual_edge_index,
            virtual_edge_attr=graph.virtual_edge_attr,
            fragment_id=graph.fragment_id,
            batch=graph.batch,
        )
        node_mask = explanation.node_mask.detach().cpu()
        if reduce == "sum_abs":
            node_masks.append(node_mask.abs().sum(dim=1).numpy())
        elif reduce == "mean_abs":
            node_masks.append(node_mask.abs().mean(dim=1).numpy())
        elif reduce == "max_abs":
            node_masks.append(node_mask.abs().max(dim=1).values.numpy())

    all_scores = np.concatenate(node_masks)
    vmin, vmax = (np.percentile(all_scores, [5, 95]) if use_percentile
                  else (all_scores.min(), all_scores.max()))

    fig.subplots_adjust(bottom=0.08, hspace=0.3)
    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])
    sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cm.get_cmap(cmap_name))
    cb = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Atom importance", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="500", y=1.01)

    plt.savefig("atom_heatmap_grid.png", bbox_inches="tight", dpi=150)
    plt.show()



from src.data.features_graph import (
    ALL_ATOM_FEATURES,
    CategoricalFeature,
    NumericFeature,
)

pt = Chem.GetPeriodicTable()

def choice_label(feature_name, value):
    if feature_name == "atomic_num":
        return f"{pt.GetElementSymbol(int(value))} ({int(value)})"
    if feature_name == "formal_charge":
        return f"{int(value):+d}"
    return str(value).split(".")[-1]

def build_atom_feature_index(atom_features):
    rows = []
    blocks = []
    col = 0

    for feature_name in atom_features:
        spec = ALL_ATOM_FEATURES[feature_name]
        start = col

        if isinstance(spec, CategoricalFeature):
            for choice in spec.choices:
                rows.append({
                    "column": col,
                    "feature_group": feature_name,
                    "encoded_feature": f"{feature_name}={choice_label(feature_name, choice)}",
                })
                col += 1

            if spec.include_unknown:
                rows.append({
                    "column": col,
                    "feature_group": feature_name,
                    "encoded_feature": f"{feature_name}=unknown",
                })
                col += 1

        elif isinstance(spec, NumericFeature):
            rows.append({
                "column": col,
                "feature_group": feature_name,
                "encoded_feature": feature_name,
            })
            col += 1

            if spec.include_missing:
                rows.append({
                    "column": col,
                    "feature_group": feature_name,
                    "encoded_feature": f"{feature_name}=missing",
                })
                col += 1

        blocks.append({
            "feature_group": feature_name,
            "start_col": start,
            "end_col": col - 1,
            "n_columns": col - start,
        })

    return pd.DataFrame(rows), pd.DataFrame(blocks)

def plot_node_feature_importance(
    explanation,
    graph,
    feature_index,
    top_n=25,
    figsize=(16, 7),
):
    feature_index = feature_index.copy()

    node_mask = explanation.node_mask.detach().cpu().numpy()
    x = graph.x.detach().cpu().numpy()

    feature_index["mask_sum"] = node_mask.sum(axis=0)
    feature_index["active_mask_sum"] = (node_mask * (x != 0)).sum(axis=0)
    feature_index["weighted_sum"] = (node_mask * np.abs(x)).sum(axis=0)

    grouped_importance = (
        feature_index
        .groupby("feature_group", sort=False)
        .agg(
            importance=("active_mask_sum", "sum"),
            mean_mask=("mask_sum", "mean"),
            n_columns=("column", "count"),
        )
        .reset_index()
    )

    group_plot_df = grouped_importance.sort_values("importance")
    encoded_plot_df = feature_index.sort_values("active_mask_sum").tail(top_n)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].barh(group_plot_df["feature_group"], group_plot_df["importance"])
    axes[0].set_xlabel("Grouped GNNExplainer importance")
    axes[0].set_ylabel("Atom feature group")
    axes[0].set_title("Importance by feature group")

    axes[1].barh(encoded_plot_df["encoded_feature"], encoded_plot_df["active_mask_sum"])
    axes[1].set_xlabel("GNNExplainer importance")
    axes[1].set_ylabel("Encoded atom feature")
    axes[1].set_title(f"Top {top_n} encoded node features")

    plt.tight_layout()
    plt.show()

    return feature_index, grouped_importance

from torch_geometric.data import Batch

def predict_with_atom_mask(model, graph, atom_mask, baseline_x):
    masked_graph = graph.clone()

    keep = atom_mask[:, None]

    masked_graph.x = torch.where(
        keep,
        graph.x,
        baseline_x,
    )

    # Important: full model expects batched graph-level metadata.
    if hasattr(masked_graph, "batch"):
        del masked_graph.batch

    batch = Batch.from_data_list([masked_graph]).to(graph.x.device)

    with torch.no_grad():
        return model(batch).view(-1)[0].item()
    

def atom_shapley_random(model, graph, n_samples=500, p=0.5):
    model.eval()

    num_atoms = graph.x.size(0)

    # Simple baseline: all-zero atom features
    # Better later: dataset mean atom features
    baseline_x = torch.zeros_like(graph.x)

    shapley = torch.zeros(num_atoms, device=graph.x.device)
    counts = torch.zeros(num_atoms, device=graph.x.device)

    for _ in range(n_samples):
        # Random coalition S
        coalition = torch.rand(num_atoms, device=graph.x.device) < p

        for atom_idx in range(num_atoms):
            if coalition[atom_idx]:
                continue

            without_i = coalition.clone()
            with_i = coalition.clone()
            with_i[atom_idx] = True

            pred_without = predict_with_atom_mask(
                model, graph, without_i, baseline_x
            )

            pred_with = predict_with_atom_mask(
                model, graph, with_i, baseline_x
            )

            marginal_effect = pred_with - pred_without

            shapley[atom_idx] += marginal_effect
            counts[atom_idx] += 1

    shapley = shapley / counts.clamp_min(1)

    return shapley

def draw_atom_shapley_heatmap_gaussian(
    smiles,
    shapley_scores,
    cmap_name="coolwarm",
    size=(700, 500),
    sigma_frac=0.07,
    alpha=0.70,
    use_percentile=True,
    percentile=95,
    title=None,
    save_path=None,
):
    """
    Draw a signed atom-level Shapley heatmap.

    Positive scores increase the model prediction.
    Negative scores decrease the model prediction.
    For log10c, positive means higher predicted concentration.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    rdDepictor.Compute2DCoords(mol)

    if isinstance(shapley_scores, torch.Tensor):
        atom_scores = shapley_scores.detach().cpu().numpy()
    else:
        atom_scores = np.asarray(shapley_scores)

    atom_scores = atom_scores.astype(float).reshape(-1)

    if atom_scores.shape[0] != mol.GetNumAtoms():
        raise ValueError(
            f"Expected {mol.GetNumAtoms()} Shapley scores, "
            f"got {atom_scores.shape[0]}."
        )

    # Render molecule.
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(*size)
        opts = drawer.drawOptions()
        opts.useBWAtomPalette()
        opts.padding = 0.1
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        import cairosvg
        png_data = cairosvg.svg2png(
            bytestring=drawer.GetDrawingText().encode(),
            output_width=size[0],
            output_height=size[1],
        )
        mol_img = np.array(Image.open(BytesIO(png_data)).convert("RGBA")).astype(float) / 255.0
    except ImportError:
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().useBWAtomPalette()
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        mol_img = np.array(Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")).astype(float) / 255.0

    H, W = size[1], size[0]

    # Atom coordinates in rendered image space.
    drawer_tmp = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer_tmp.drawOptions().useBWAtomPalette()
    drawer_tmp.DrawMolecule(mol)
    drawer_tmp.FinishDrawing()

    atom_positions = []
    for atom_idx in range(mol.GetNumAtoms()):
        pt = drawer_tmp.GetDrawCoords(atom_idx)
        atom_positions.append((pt.x, pt.y))

    # Build signed Gaussian field.
    sigma = sigma_frac * W
    xs = np.linspace(0, W - 1, W)
    ys = np.linspace(0, H - 1, H)
    xx, yy = np.meshgrid(xs, ys)

    field = np.zeros((H, W), dtype=float)

    for score, (px, py) in zip(atom_scores, atom_positions):
        gaussian = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2))
        field += float(score) * gaussian

    if use_percentile:
        limit = np.percentile(np.abs(field), percentile)
    else:
        limit = np.max(np.abs(field))

    if limit == 0:
        limit = 1e-6

    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = cm.get_cmap(cmap_name)

    heat_rgba = cmap(norm(field))

    # Make weak regions transparent, strong positive/negative regions opaque.
    strength = np.clip(np.abs(field) / limit, 0, 1)
    heat_rgba[..., 3] = alpha * strength

    fg_alpha = heat_rgba[..., 3:4]
    composite = heat_rgba[..., :3] * fg_alpha + mol_img[..., :3] * (1 - fg_alpha)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(size[0] / 100, (size[1] + 60) / 100),
        gridspec_kw={"height_ratios": [size[1], 60], "hspace": 0.03},
        dpi=100,
    )

    ax_img, ax_cb = axes

    ax_img.imshow(composite, origin="upper", interpolation="bilinear")
    ax_img.axis("off")
    ax_img.set_title(title or smiles, fontsize=13, pad=10)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_label("Shapley contribution to prediction", fontsize=10, labelpad=4)
    cb.ax.tick_params(labelsize=9)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()

    return atom_scores

def predict_graph(model, graph, device):
    g = graph.clone()

    if hasattr(g, "batch") and g.batch is not None:
        del g.batch

    batch = Batch.from_data_list([g]).to(device)

    model.eval()
    with torch.no_grad():
        return model(batch).view(-1).item()

def cat_pred_sweep(model, graph, category, categorical_encoder, device):
    encoder = categorical_encoder[category]
    decoder = {v: k for k, v in encoder.items()}

    base_pred = predict_graph(model, graph, device)

    raw = getattr(graph, category)
    base_id = int(raw.item() if isinstance(raw, torch.Tensor) else raw)    
    base_name = decoder[base_id]

    rows = []

    for name, id in encoder.items():
        g = graph.clone()

        if hasattr(g, "batch") and g.batch is not None:
            del g.batch

        new_val = torch.tensor(id, dtype=torch.long, device=g.x.device)
        setattr(g, category, new_val)

        pred_log10c = predict_graph(model, g, device)

        rows.append({
                    category : name,
                    f"{category}_id" : id,
                    "pred_log10c" : pred_log10c,
                    "pred_conc_same_unit" : 10 ** pred_log10c,
                    "delta_log10c_vs_original" : pred_log10c - base_pred,
                    f"original_{category}" : base_name,
                })

    return (
        pd.DataFrame(rows)
        .sort_values("pred_log10c")
        .reset_index(drop=True)
    )

import copy
from torch import nn


class ZeroTaxidEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, data):
        # infer batch size from graph batch if available
        if hasattr(data, "batch") and data.batch is not None:
            batch_size = int(data.batch.max().item()) + 1
        else:
            batch_size = data.y.view(-1).numel() if hasattr(data, "y") else 1

        device = data.x.device
        return torch.zeros(batch_size, self.output_dim, device=device)

def plot_species_sweep_muted(
    model,
    graph,
    categorical_encoder,
    device,
    df=None,
    figsize=(9, 6),
):

    model.eval()

    # Make no-taxid copy
    model_no_taxid = copy.deepcopy(model)
    taxid_encoder = model_no_taxid.meta_encoder.pretrained_taxid_encoder
    taxid_dim = taxid_encoder.output_dim

    model_no_taxid.meta_encoder.pretrained_taxid_encoder = (
        ZeroTaxidEncoder(taxid_dim).to(device)
    )
    model_no_taxid.eval()

    # Run sweeps
    sweep_with_taxid = cat_pred_sweep(
        model, graph, "species_group", categorical_encoder, device
    )

    sweep_no_taxid = cat_pred_sweep(
        model_no_taxid, graph, "species_group", categorical_encoder, device
    )

    comparison = sweep_with_taxid.merge(
        sweep_no_taxid,
        on="species_group",
        suffixes=("_with_taxid", "_no_taxid"),
    )

    comparison = comparison.sort_values("pred_log10c_with_taxid")

    # Pull molecule / endpoint info
    row_id = int(graph.row_id.item()) if hasattr(graph, "row_id") else None

    smiles = getattr(graph, "smiles", "unknown molecule")
    endpoint = None
    chemical_name = None

    if df is not None and row_id is not None:
        endpoint = df.iloc[row_id].get("endpoint", None)
        chemical_name = df.iloc[row_id].get("chemical_name", None)

    if endpoint is None and hasattr(graph, "endpoint"):
        endpoint_decoder = {
            v: k for k, v in categorical_encoder["endpoint"].items()
        }
        endpoint = endpoint_decoder.get(int(graph.endpoint.item()), "unknown endpoint")

    title_parts = []
    if chemical_name:
        title_parts.append(str(chemical_name))
    else:
        title_parts.append(str(smiles))

    if endpoint:
        title_parts.append(f"endpoint: {endpoint}")

    title = " | ".join(title_parts)

    # Plot
    ax = comparison.plot(
        x="species_group",
        y=["pred_log10c_with_taxid", "pred_log10c_no_taxid"],
        kind="barh",
        figsize=figsize,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted log10 concentration")
    ax.set_ylabel("Counterfactual species group")
    ax.legend(["With taxid", "Taxid muted"])

    plt.tight_layout()
    plt.show()

    return comparison