# Project Structure Review

## Overall impression

The project already has a meaningful split between data preparation, models, and helper code under `src/`, and the newer notebooks are noticeably more intentional than the original main notebook. That is a strong base. The main thing missing right now is a sharper boundary between:

- reusable pipeline code
- notebook orchestration and exploration
- experiment outputs and scratch files

The current structure works for active thesis development, but it still feels partly like a research workspace and partly like a reusable codebase. That is normal at this stage, but it also explains why some responsibilities are a bit mixed.

## What is already good

- `src/data`, `src/model`, and `src/utils` are a sensible first-level split.
- The newer notebooks are much clearer than the original `pybook.ipynb`. In particular, `pybook_data_analysis.ipynb` and `pybook_model_training.ipynb` follow a readable step-by-step narrative.
- Important notebook settings are kept near the cells that use them, which is good for experimentation.
- The model code is already separated into distinct components: graph encoders, metadata encoder, and the combined prediction model.
- The source files are still small enough to refactor safely. Nothing feels so large that restructuring would be painful.

## Main structure issues

### 1. Root directory is doing too many jobs

Right now the repository root contains notebooks, source code, generated figures, a scratch script, and the dataset folder. That makes it harder to see what is canonical.

Examples:

- `pybook.ipynb`
- `pybook_data_analysis.ipynb`
- `pybook_model_training.ipynb`
- `testingbook.ipynb`
- `testscrpit.py`
- `metal_distribution.png`

This makes the project feel more experimental than it probably needs to.

### 2. `src/utils` is a catch-all

`src/utils/plotting.py` and `src/utils/splitter.py` are useful, but they are not really generic utilities. They are domain-specific parts of the data and experiment pipeline.

- `splitter.py` is more like dataset splitting logic than a generic utility.
- `plotting.py` is more like analysis/visualization support than a generic utility.

When modules land in `utils`, it usually means "we have not decided where this really belongs yet."

### 3. Some `src/data` modules mix multiple responsibilities

`src/data/dataMol.py` currently combines:

- CSV loading
- molecule inspection helpers
- optional DeepChem featurization

That is three different responsibilities in one file.

`src/data/dataMeta.py` currently combines:

- taxonomy encoding
- loading taxonomy columns
- assembling final graph objects with labels and metadata

That is also more than one layer of concern.

### 4. The model package also contains training code

`src/model/eval.py` is not really a model definition file. It is training and evaluation workflow code.

Structurally, training loops usually belong in something like:

- `src/training/`
- `src/train/`
- or `src/experiments/`

Keeping training logic inside `model/` blurs the distinction between architecture and training procedure.

### 5. There are signs of old and new approaches coexisting

This is the clearest structural theme in the repo.

- `pybook.ipynb` is more exploratory and manual.
- `pybook_model_training.ipynb` is cleaner and more pipeline-like.
- `src/data/dataMol.py` still contains a DeepChem-based `featurize()` path even though the main notebooks now use PyG `from_smiles`.
- `src/model/eval.py` still imports DeepChem metrics and has a `predict()` helper that looks out of sync with the current PyTorch Geometric training flow.

This is not just a code cleanliness issue. It makes it harder to tell what the "real" pipeline is.

### 6. Naming is a bit inconsistent

The current file names mix styles and meanings:

- `dataMol.py`
- `dataMeta.py`
- `modelFull.py`
- `eval.py`

These names are understandable once you know the project, but they are not especially descriptive to a new reader. They also do not follow typical snake_case Python file naming.

## Specific observations from the current files

### `src/data/dataMol.py`

Good:

- centralizes basic dataframe loading
- keeps chemistry-specific helpers close together

Needs improvement:

- mixes IO, chemistry rules, and featurization
- contains an optional DeepChem path that does not look like the main direction of the project

### `src/data/dataMeta.py`

Good:

- taxonomy handling is clearly separated from model code
- `build_graph_features()` is a useful bridge between tabular data and graph objects

Needs improvement:

- taxonomy encoding and graph assembly are related, but still distinct responsibilities
- `load_taxonomy_dataframe()` suggests a more reusable API, but the notebooks still partly reconstruct this logic manually

### `src/data/preprocess.py`

Good:

- the file has a focused purpose
- molecule cleaning operations are grouped together

Needs improvement:

- target transformation lives beside structural cleaning, which is workable but conceptually different
- this module feels like it wants to become a clearer "cleaning + target prep" pipeline module

### `src/model/`

Good:

- separate files for `GCN`, `GIN`, `AttentiveFP`, metadata encoder, and combined model make sense
- the composition pattern in `ToxicityModel` is clean

Needs improvement:

- `eval.py` belongs outside the model architecture package
- `modelFull.py` and `modelMeta.py` could be named more clearly

### `src/utils/splitter.py`

Good:

- important experimental logic is reusable instead of being trapped in notebooks

Needs improvement:

- this is really part of dataset preparation / splitting, not a generic utility
- the file contains leftover imports and experimental residue, which suggests it evolved organically

### `src/utils/plotting.py`

Good:

- useful analysis helpers are centralized
- plotting functions support both exploration and training review

Needs improvement:

- analysis plots and training plots are mixed together
- `plot_metals()` writes `metal_distribution.png` directly into the project root, which creates output clutter

## Notebook review

### What is working well

- `pybook_data_analysis.ipynb` is the clearest exploration notebook.
- `pybook_model_training.ipynb` is the clearest modeling notebook.
- Both newer notebooks use markdown structure well and read more like reproducible reports than scratchpads.

### What is still unclear

- `pybook.ipynb` is still the original "everything notebook," but it overlaps with the newer notebooks.
- `testingbook.ipynb` and `testscrpit.py` look like scratch artifacts rather than stable project assets.
- Because the notebooks live at the root, it is not obvious which notebook a new reader should start with.

## Recommended target structure

I would move toward something like this:

```text
gnn-thesis/
├── Data/
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── archive/
│       ├── pybook.ipynb
│       └── testingbook.ipynb
├── outputs/
│   ├── figures/
│   └── reports/
├── src/
│   ├── data/
│   │   ├── io.py
│   │   ├── cleaning.py
│   │   ├── taxonomy.py
│   │   ├── graph_building.py
│   │   └── splitting.py
│   ├── models/
│   │   ├── gcn.py
│   │   ├── gin.py
│   │   ├── attentive_fp.py
│   │   ├── meta_encoder.py
│   │   └── toxicity_model.py
│   ├── training/
│   │   ├── loops.py
│   │   └── metrics.py
│   └── visualization/
│       ├── data_plots.py
│       └── training_plots.py
├── tests/
└── README.md
```

## How I would map the current files into that structure

- `src/data/dataMol.py`
  - split into `src/data/io.py`, `src/data/cleaning.py`, and possibly `src/data/featurization.py` if the DeepChem path is still needed
- `src/data/dataMeta.py`
  - split into `src/data/taxonomy.py` and `src/data/graph_building.py`
- `src/data/preprocess.py`
  - fold into `src/data/cleaning.py` or keep as `src/data/preprocessing.py`
- `src/utils/splitter.py`
  - move to `src/data/splitting.py`
- `src/utils/plotting.py`
  - move to `src/visualization/` and split by purpose if it grows
- `src/model/eval.py`
  - move to `src/training/loops.py`
- `src/model/modelGCN.py`
  - rename to `src/models/gcn.py`
- `src/model/modelGIN.py`
  - rename to `src/models/gin.py`
- `src/model/modelAFP.py`
  - rename to `src/models/attentive_fp.py`
- `src/model/modelMeta.py`
  - rename to `src/models/meta_encoder.py`
- `src/model/modelFull.py`
  - rename to `src/models/toxicity_model.py`

## Recommended priorities

If you want to improve clarity without doing a disruptive rewrite, I would prioritize changes in this order:

1. Make the notebook story clear.
   - Put notebooks in a `notebooks/` folder.
   - Keep `pybook_data_analysis.ipynb` and `pybook_model_training.ipynb` as the main notebooks.
   - Move `pybook.ipynb` and `testingbook.ipynb` to an `archive/` or `scratch/` area.

2. Remove the catch-all feeling from `utils`.
   - Move splitting into `data/`.
   - Move plotting into `visualization/`.

3. Separate architecture code from training code.
   - Move `eval.py` out of `model/`.

4. Split mixed-responsibility data modules.
   - Separate loading, cleaning, taxonomy encoding, and graph building into clearer files.

5. Clean up old experimental paths.
   - Decide whether DeepChem is still part of the intended pipeline.
   - If not, remove or archive the leftover DeepChem-specific code paths.

6. Add a small `tests/` folder.
   - Even a few tests for preprocessing, splitting, and graph construction would make later refactors much safer.

## Final opinion

The project structure is already better than many thesis repositories because there is a real attempt to move logic into `src/` and to keep notebooks readable. The biggest issue is not that the structure is bad. It is that the codebase currently reflects multiple stages of the project at once:

- early exploration
- emerging reusable pipeline code
- newer, more disciplined notebook workflows

That makes the repository slightly harder to read than it needs to be.

My main recommendation is to treat the current structure as a strong draft, then make the responsibilities more explicit:

- notebooks for analysis and orchestration
- `src/data` for dataset logic
- `src/models` for architectures only
- `src/training` for fit/eval loops
- `src/visualization` for plots
- `outputs/` for generated artifacts

That would make the project feel much clearer, more reproducible, and more appropriate for a thesis codebase without changing the overall workflow you already have.
