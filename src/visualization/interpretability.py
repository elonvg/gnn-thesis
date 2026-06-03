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