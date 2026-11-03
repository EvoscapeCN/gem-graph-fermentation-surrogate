# Data format

All fermentation inputs live in one directory, `${MNM_DATA_DIR}` (default
`data/synthetic`). The configs expect these file names:

| File | Read by | Purpose |
|---|---|---|
| `metadata.csv` | `dataset.fermentation` | one row per fermentation run |
| `time_series.csv` | `dataset.fermentation` | sensor streams and assays, long format |
| `gene_expression.csv` | `dataset.fermentation` | RNA-seq snapshots, one column per snapshot |
| `culture_medium_reference.csv` | `dataset.fermentation` | reference recipe for the medium log-ratio |
| `flux_corpus.parquet` (or `.csv`) | `dataset.flux_corpus` | FBA flux vectors for pretraining |

The GEM lives in `data/gem/iyli21_modified.xml`, which every config reads; `scripts/build_gem.py` builds it from iYli21 (see the README). `scripts/make_synthetic_data.py` writes all
five files with random content in this format.

The code does not rescale anything. In the paper, sensor streams, assays and
cumulative targets were scaled to roughly `[0, 1]` beforehand with per-metric
global 1st/99th-percentile anchors, and rates are clipped to +-5 (scaled
units per hour) during integration, so inputs should be on a comparable scale.

## `metadata.csv`

| Column | Type | Meaning |
|---|---|---|
| `ferm_id` | str | Unique run identifier; the unit of every train/validation/test split. |
| `task`, `reactor_id` | str | Together they link the run to its expression columns `<task>-<reactor_id>-<rna_time>`. `reactor_id` must match `R<digits>` (e.g. `R03`); `task` may contain `-`. |
| `rna_times` | str | Comma-separated sampling times (h) of the run's RNA snapshots, e.g. `"12, 24, 36"`. Times without an expression column are ignored; runs with no matching snapshot are skipped. |
| `initial_od` | float | Inoculation density, used as a raw number. |
| `culture_medium_change` | str | Deviations from the reference medium, `"NAME: value, NAME: value"` (`;` also accepted as separator). Components not listed keep their reference amount; empty means the reference medium. Every name must appear in `culture_medium_reference.csv`. |
| `strain_id` | str, optional | Only read when `dataset.strain_ids` is set (runs of other strains are dropped). |

The run-level condition vector is built from `dataset.initial_condition_fields`
(default `[initial_od, culture_medium_change]`): `initial_od` as one number
and the medium as `log((c + 1e-6) / (c0 + 1e-6))` per reference component.
Any other numeric metadata column can be added to that list. Other columns are
ignored.

## `culture_medium_reference.csv`

| Column | Type | Meaning |
|---|---|---|
| `component` | str | Medium component name (13 in the paper). The row order fixes the order of the log-ratio features. |
| `reference` | float | Amount in the reference recipe (`c0`). |

## `time_series.csv`

Long format: one row per (run, time point). Sensor rows and assay rows may be
interleaved.

| Column | Type | Meaning |
|---|---|---|
| `ferm_id` | str | Run identifier (as in `metadata.csv`). |
| `time` | float | Hours since inoculation. |
| `is_original_assay` | bool | `True` on rows that carry the offline assays. These rows define the run's assay time grid `t_1 < ... < t_T` (prediction and loss points). If the column is absent, every row is an assay row. |
| assay columns | float | Named by `dataset.assay_metrics`. Cumulative quantities; `NaN` = not measured. Paper configs: `cumulated_o2_consumption`, `cumulated_sugar_consumption`, `interpolated_biomass`, `carotenoids`, `astaxanthin`. |
| sensor columns | float | Named by `dataset.sensor_metrics` (paper configs: `time`, `agitation_speed`, `temp`, `feed_temp`, `ph`, `do`). |

For each assay interval `[t_i, t_{i+1})` (the last one is `[t_T, t_T + 1 h)`),
every sensor column is summarised over all rows of the run in that window
(sensor and assay rows alike) by four statistics: median, interquartile range,
and the slope and correlation coefficient of a linear fit against `time`. The
last interval reuses the slope and correlation of the one before. A window with
no data gives zeros. This yields `sensor_stats [T, n_sensor, 4]`. In the
paper's data the table held one row per assay time, so the median is the
reading at `t_i` and the other three statistics are zero.

Assay values are used only as loss targets and as the starting point of the
integration of the four specific-rate readouts (each readout starts at its first
valid, non-zero measurement; biomass is integrated from zero). A cumulative
measurement that falls below the previous valid one is excluded from the loss.

## `gene_expression.csv`

| Column | Type | Meaning |
|---|---|---|
| `Gene_ID` | str | Gene identifier (any naming). |
| `XML_ID` | str | The gene's id in the SBML model, with or without the SBML `G_` prefix; empty for genes without a GPR association. |
| `<task>-<reactor_id>-<rna_time>` | float | One column per RNA snapshot: log1p-normalised expression (the paper used DESeq2-style size factors followed by log1p). |

Every gene in the file is used by the whole-transcriptome readout features.
The GPR genes of the SBML model are matched through `XML_ID`; a GPR gene with
no row in the file is set to the constant 1.0. From the training split only,
the dataset computes each gene's mean raw-scale expression (for the GPR
relative-expression ratio, clipped to `[0, 5]`), per-gene means and standard
deviations (SD floor 0.05) and, when `expression_pca_components > 0`, the
whitened principal components of the standardised snapshots.

## Flux corpus (`flux_corpus.parquet` / `.csv`)

One row per flux vector.

| Column | Type | Meaning |
|---|---|---|
| one column per SBML reaction id | float | Net flux (mmol gDW^-1 h^-1); negative values run the reverse direction. All reactions of the model must be present. |
| `time` | float | Time point (h) the vector was sampled for; used for time stratification (`time_stage_boundaries`, right-inclusive). |
| `near_opt_frac` | float, optional | Fraction of the biomass optimum the vector was constrained to; rows outside `[min_near_opt_frac, max_near_opt_frac]` are dropped. |

Other columns are ignored. For each half-node the dataset uses `max(v, 0)`
(forward) or `max(-v, 0)` (reverse) as the masked-reconstruction target, and
computes ten robust statistics of the corpus (share of near-zero samples,
median, MAD, IQR, medcouple, trimmed kurtosis, L-moments `L1`, `L2`,
`L3/L2`, `L4/L2` of `log10(|v| + 1e-4)`), robust-scaled across half-nodes, as
the input of the flux encoder.

The paper's corpus has 40,250 vectors over 15 time points under time-dependent
exchange bounds. The demo corpus is sampled with
cobrapy's OptGP sampler from the GEM built by `scripts/build_gem.py` at the SBML's own bounds with
biomass constrained to at least 95% of its optimum; its `time` values are
assigned at random.
