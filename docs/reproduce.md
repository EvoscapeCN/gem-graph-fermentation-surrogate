# Paper tables and figures -> configs -> commands

With `MNM_DATA_DIR` pointing at data in the format of `data_format.md`, the
commands below run the same configurations, splits, seeds and evaluation as the
paper. By default they run on the synthetic demo data in `data/synthetic/`.

Common protocol: 5 folds by run (`KFold(shuffle=True, random_state=42)` over
the sorted run ids; one fold is the test set and `n_trainval // 5` of the
remaining runs are validation), seeds 42-46 (each fold trains with seed
`seed + fold`), model selection by best validation loss (`best.pt`), and one
NumPy evaluation (`src/mnm/evaluation.py`) for every method. The paper's runs
are those of a single strain; restrict the metadata file to the runs you want,
or set `dataset.strain_ids`.

All experiments write `outputs/experiments/<experiment>/results.json`;
`run_stats.py` reads every experiment present there.

| Paper item | Configs | Command |
|---|---|---|
| Main comparison table | `finetune_main.yaml` (Ours), `ablations/no_biomass_head.yaml` (GNN), Ridge / XGBoost (`flat_ml` methods) | `python scripts/run_experiment.py --yaml configs/experiments/main_comparison.yaml` |
| Run-level contrasts table, tail-risk statistics, power | (results of the above) | `python scripts/run_stats.py --results-dir outputs/experiments --out outputs/stats.json --macros outputs/macros.tex` |
| Ablation table, per-fold ablation table | `ablations/*.yaml` | `python scripts/run_experiment.py --yaml configs/experiments/ablations.yaml` (the -BiomassHead row is `gnn` in `main_comparison`) |
| Structural-controls tables (full and per-readout) | `controls/*.yaml` (randomisation seeds per model seed in the manifest) | `python scripts/run_experiment.py --yaml configs/experiments/controls.yaml` |
| Baselines on GPR genes / 16 PCs (appendix) | `flat_ml` methods with `use_gpr_genes` / `gene_input: pcs` | `python scripts/run_experiment.py --yaml configs/experiments/input_matched_baselines.yaml` |
| Flux-projection reuse, FiLM, causal-kernel rerun (appendix, Sec. 4.3) | `appendix/*.yaml` | `python scripts/run_experiment.py --yaml configs/experiments/appendix.yaml` |
| Kernel bandwidth and future-weight figures (Sec. 4.3, appendix) | `finetune_main.yaml` | `python scripts/kernel_stats.py --results outputs/experiments/main_comparison/results.json --base main` |
| Trajectory figure (appendix) | `finetune_main.yaml` | `python scripts/export_trajectories.py --results outputs/experiments/main_comparison/results.json --yaml configs/experiments/main_comparison.yaml --base main --out outputs/figures/trajectories` then `python scripts/plot_trajectories.py outputs/figures/trajectories_selected.npz outputs/figures/trajectories` |
| Pretrained encoder | `pretrain.yaml` | `python scripts/train.py --config configs/pretrain.yaml --output-dir outputs/pretrain` (reads a flux corpus in the format of `data_format.md`; the released `checkpoints/pretrained_encoder.pt` is the epoch-20 checkpoint of this configuration) |
| Model size (appendix) | `finetune_main.yaml` | printed by every training run (`task parameters: ...`) |

To finetune from your own pretraining run instead of the released encoder, set
`task.model.checkpoint_path` (e.g. `--set task.model.checkpoint_path=outputs/pretrain/checkpoints/epoch_0020.pt`).

Useful `run_experiment.py` options: `--methods a,b` (subset), `--folds 0,1`,
`--n-folds`, `--set key=value` (applied to every model config, e.g.
`trainer.device=cpu`), `--resume`.
