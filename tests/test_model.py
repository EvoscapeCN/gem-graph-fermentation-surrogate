"""Graph size, parameter count and strict loading of the released encoder
for the main configuration (on the committed synthetic data)."""
import os

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_mfg_size():
    from mnm.graph import build_mfg
    g = build_mfg(os.path.join(REPO, "data/gem/iyli21_modified.xml"))
    assert g["num_nodes"] == 4638
    assert g["edge_index"].shape[1] == 172061


def test_main_model_loads_encoder_and_parameter_count(monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.setenv("MNM_DATA_DIR", os.path.join(REPO, "data/synthetic"))
    import mnm  # noqa: F401
    from mnm.registry import build_from_config, load_config, resolve_variables
    cfg = resolve_variables(load_config("configs/finetune_main.yaml"))
    cfg["dataset"].update(k_fold=True, split_id=0, n_folds=2)
    built = build_from_config({k: v for k, v in cfg.items() if k != "trainer"})
    ds, task = built["dataset"], built["task"]
    batch = ds.collate([ds[ds._split_indices["train"][0]]])
    with torch.no_grad():
        losses, _ = task.compute_batch(batch)
    assert torch.isfinite(losses["loss"])
    # encoder weights equal the released checkpoint (strict load of gnn / adapter / metabolites)
    ck = torch.load("checkpoints/pretrained_encoder.pt", map_location="cpu")["task"]
    sd = task.state_dict()
    for k, v in ck.items():
        if k.startswith(("model.gnn_backbone.layers", "model.gnn_backbone.norms", "model.gnn_backbone.final_proj",
                         "model.adapter.", "model.metabolites_encoder.")):
            assert torch.equal(sd[k], v), k
    # GEM built by scripts/build_gem.py (1,069 GPR genes, 128-dim gene embeddings):
    # 1,971,403 trainable parameters,
    # 548,089 while the pretrained modules are frozen
    assert sum(p.numel() for p in task.parameters()) == 1971403
    task.model.freeze_pretrained_modules()
    assert sum(p.numel() for p in task.parameters() if p.requires_grad) == 548089
