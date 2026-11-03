#!/usr/bin/env python
"""Build the modified iYli21 GEM that every config reads (data/gem/iyli21_modified.xml).

iYli21 (Guo et al., 2022, Comput. Struct. Biotechnol. J., doi:10.1016/j.csbj.2022.05.018)
is distributed under CC BY-NC-ND 4.0, so this repository ships the modifications
(data/gem/iyli21_patch.json) instead of the modified model. Download the model from the
article's supplementary material (Supplementary data 6, mmc6.xml), then:

    python scripts/build_gem.py --source path/to/mmc6.xml

The patch removes ten alternative biomass and lipid-pool reactions, adds the astaxanthin pathway, the carotenoid pool and the carotenoid terms of
biomass_C, adjusts some reaction bounds and GPR annotations, and fixes the order of
reactions, metabolites and genes (the pretrained metabolite embeddings are indexed by it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import cobra

REPO = Path(__file__).resolve().parent.parent
PATCH = REPO / "data" / "gem" / "iyli21_patch.json"
OUT = REPO / "data" / "gem" / "iyli21_modified.xml"
SOURCE_SHA256 = "6974b7588f2a6c60ba2cde2f26e20d3aba1334d0d501572bf01cee47eda86631"


def rename_gene(gid: str) -> str:
    """iYli21 gene ids (YALI1C08548g) -> ids used here (YALI1_C08548g)."""
    return gid[:5] + "_" + gid[5:]


def apply_patch(model: cobra.Model, patch: dict) -> cobra.Model:
    cobra.manipulation.rename_genes(model, {g.id: rename_gene(g.id) for g in model.genes})
    for cid, name in patch["compartments"].items():
        model.compartments = {**model.compartments, cid: name}
    model.remove_reactions([model.reactions.get_by_id(k) for k in patch["reactions_remove"]])
    model.remove_metabolites([model.metabolites.get_by_id(k) for k in patch["metabolites_remove"]])
    for mid, d in patch["metabolite_updates"].items():
        met = model.metabolites.get_by_id(mid)
        met.name, met.formula, met.charge, met.compartment = d["name"], d["formula"], d["charge"], d["compartment"]
    model.add_metabolites([cobra.Metabolite(d["id"], formula=d["formula"], name=d["name"], charge=d["charge"],
                                            compartment=d["compartment"]) for d in patch["metabolites_add"]])
    new = []
    for d in patch["reactions_add"]:
        r = cobra.Reaction(d["id"], name=d["name"], subsystem=d["subsystem"],
                           lower_bound=d["lower_bound"], upper_bound=d["upper_bound"])
        new.append((r, d))
    model.add_reactions([r for r, _ in new])
    for r, d in new:
        r.add_metabolites({model.metabolites.get_by_id(k): v for k, v in d["metabolites"].items()})
        r.gene_reaction_rule = d["gene_reaction_rule"]
    for rid, u in patch["reaction_updates"].items():
        r = model.reactions.get_by_id(rid)
        if "metabolites" in u:
            r.subtract_metabolites(dict(r.metabolites))
            r.add_metabolites({model.metabolites.get_by_id(k): v for k, v in u["metabolites"].items()})
        if "lower_bound" in u:
            r.bounds = (u["lower_bound"], u["upper_bound"])
        if "gene_reaction_rule" in u:
            r.gene_reaction_rule = u["gene_reaction_rule"]
    keep = set(patch["order"]["genes"])
    cobra.manipulation.remove_genes(model, [g for g in model.genes if g.id not in keep], remove_reactions=False)
    for g in model.genes:
        g.name = patch["gene_names"].get(g.id, g.name)
    order = patch["order"]
    for attr in ("metabolites", "reactions", "genes"):
        rank = {k: i for i, k in enumerate(order[attr])}
        items = getattr(model, attr)
        assert set(rank) == {x.id for x in items}, attr
        items.sort(key=lambda x: rank[x.id])
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="iYli21 SBML (mmc6.xml)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    logging.getLogger("cobra").setLevel(logging.ERROR)
    digest = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        print(f"warning: {args.source} differs from the mmc6.xml the patch was made for (sha256 {digest})")
    model = apply_patch(cobra.io.read_sbml_model(args.source), json.loads(PATCH.read_text()))
    cobra.io.write_sbml_model(model, args.out)
    print(f"{len(model.reactions)} reactions, {len(model.metabolites)} metabolites, {len(model.genes)} genes "
          f"-> {args.out} (biomass optimum {model.slim_optimize():.4f})")


if __name__ == "__main__":
    main()
