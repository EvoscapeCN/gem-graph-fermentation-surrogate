# GEM-graph surrogate for fermentation dynamics

Code for the NeurIPS 2026 paper

> **Toward in Silico Strain Evaluation: A Multimodal Surrogate for Fermentation Dynamics with Metabolic Graph Pretraining**
> Yunxiao Li¹, Difeng Gao¹, Yubin Zheng¹, Jijiao Zeng¹²
> ¹Evoscape Bioscience, Hangzhou, China; ²Institute of Health and Medicine, Hefei Comprehensive National Science Center, Hefei, China

The model is a spatio-temporal GNN over a Mass Flow Graph (MFG) built from a
genome-scale metabolic model (GEM; a modified iYli21). Each reaction contributes
a forward and a reverse half-node; an edge joins two half-nodes when the first
produces a metabolite the second consumes. A 3-layer GATv2 encoder over this
graph is pretrained by masked-flux reconstruction on Flux Balance Analysis
(FBA) samples, then finetuned on fermentation runs with four modalities:
gene-expression snapshots (routed to reactions through the GPR associations,
plus whole-transcriptome readout features), online sensor streams, offline
assays (targets only) and run-level metadata. A Gaussian temporal kernel
interpolates the per-snapshot node states to each query time; a GRU emits
specific rates for oxygen, glucose, total carotenoids and astaxanthin, an
independent MLP emits the biomass rate, and the rates are integrated (forward
Euler) to cumulative trajectories trained under an RMSE loss.

## What this repository contains

| | |
|---|---|
| `src/mnm/` | MFG construction (`graph.py`), datasets (`dataset.py`), encoders and model (`layers.py`, `model.py`), pretraining and finetuning objectives (`task.py`), trainer (`train.py`), the evaluation shared by all methods (`evaluation.py`), Ridge/XGBoost features (`baselines.py`) |
| `configs/` | `pretrain.yaml`; `finetune_main.yaml` (main model); `ablations/`, `controls/`, `appendix/` (one file per table row); `experiments/` (5-fold x 5-seed manifests) |
| `scripts/` | `run_experiment.py` (k-fold x seeds driver), `run_stats.py` (run-level bootstrap statistics), `train.py` (single run, e.g. pretraining), `kernel_stats.py`, `export_trajectories.py`, `plot_trajectories.py`, `make_synthetic_data.py` |
| `checkpoints/pretrained_encoder.pt` | the pretrained encoder used for every finetuning result in the paper (weights only) |
| `data/gem/iyli21_patch.json`, `scripts/build_gem.py` | our modifications to iYli21 and the script that applies them to the published model (see [The GEM](#the-gem)) |
| `data/synthetic/` | a small synthetic dataset in the input format, generated from random curves and random expression by `scripts/make_synthetic_data.py` |
| `docs/` | `data_format.md` (every input file and column), `reproduce.md` (table/figure -> config -> command) |

## What you can do with it

* inspect the exact architecture and training procedure, e.g. the parameter
  count of the main model (1,971,403 with the GEM built here, of which 548,089
  train while the encoder is frozen);
* finetune from the released pretrained encoder on your own fermentation runs,
  after putting them in the format of `docs/data_format.md`;
* pretrain the encoder on your own flux corpus;
* run every experiment pipeline end to end on the synthetic data.

## The GEM

iYli21 (Guo et al., 2022) is distributed under CC BY-NC-ND 4.0, so this
repository ships our modifications to it, not the modified model. Download
iYli21 from the supplementary material of the article
([doi:10.1016/j.csbj.2022.05.018](https://doi.org/10.1016/j.csbj.2022.05.018),
Supplementary data 6, `mmc6.xml`) and build the GEM that every config reads:

```bash
python scripts/build_gem.py --source /path/to/mmc6.xml   # writes data/gem/iyli21_modified.xml
```

The modifications add the astaxanthin pathway, a carotenoid pool and carotenoid
terms in the biomass reaction, cap the bounds of some internal reactions, add
GPR annotations, and remove ten alternative biomass and lipid-pool reactions.
In this release the carotenoid composition (the pool reaction `xpigment` and the
carotenoid terms of `biomass_C`) and the carotenoid content of biomass are
placeholders, and strain-specific GPR entries are omitted, so results obtained
with this GEM differ slightly from those reported in the paper.

## Quick start

```bash
conda env create -f environment.yml
conda activate mnm
python scripts/build_gem.py --source /path/to/mmc6.xml

# end-to-end smoke test on CPU (synthetic data -> pretraining -> finetuning -> evaluation -> statistics)
python -m pytest tests/ -q
```

Step by step (all commands from the repository root; data paths resolve against
`MNM_DATA_DIR`, default `data/synthetic`; outputs go to `outputs/`):

```bash
# (optional) regenerate the synthetic data
python scripts/make_synthetic_data.py --out data/synthetic

# short pretraining demo on the synthetic flux corpus
python scripts/train.py --config configs/pretrain.yaml --output-dir outputs/pretrain_demo \
    --set trainer.epochs=1 --set trainer.num_workers=0 --set trainer.device=cpu

# main model and XGBoost, 2 folds x 1 seed, 1 epoch, starting from checkpoints/pretrained_encoder.pt
python scripts/run_experiment.py --yaml configs/experiments/main_comparison.yaml \
    --methods main_seed42,xgb_seed42 --n-folds 2 \
    --set trainer.epochs=1 --set trainer.device=cpu

# run-level statistics of whatever methods have results
python scripts/run_stats.py --results-dir outputs/experiments --out outputs/stats.json
```

On a GPU, drop the `--set` overrides to run the configurations as locked
(50 epochs, AMP). With your own data in another directory:

```bash
export MNM_DATA_DIR=/path/to/your/data
python scripts/run_experiment.py --yaml configs/experiments/main_comparison.yaml
```

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) (noncommercial use only).
Copyright (c) 2026 Evoscape Bioscience. `data/gem/iyli21_patch.json` modifies
iYli21 (Guo et al., 2022; CC BY-NC-ND 4.0), which is not included; please also
cite the original model.

## Citation

See [`CITATION.cff`](CITATION.cff).
