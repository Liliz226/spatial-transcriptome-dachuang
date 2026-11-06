"""Batch benchmark: run a dataset under all spatial-weight schemes and
summarise the metrics into one CSV for comparison.

Usage:
    python benchmark.py --dataset Lung5-3 --config datasets.json \
        --epochs 800 --device cuda:0 --output /path/to/out

Each run trains STAMP once per spatial weight (binary/gaussian/cosine/
combined), evaluates metrics, and appends a row to <output>/summary.csv.
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import squidpy as sq
import sctm


SPATIAL_WEIGHTS = ["binary", "gaussian", "cosine", "combined"]


def load_and_prepare(ad_path, cfg):
    """Load one h5ad and apply the preprocessing declared in cfg."""
    adata = ad.read_h5ad(ad_path)
    adata.var_names_make_unique()

    layer = cfg.get("layer")
    # Some notebooks put counts in X and copy it to a named layer.
    if cfg.get("copy_x_to_layer"):
        adata.layers[cfg["copy_x_to_layer"]] = adata.X.copy()
        layer = cfg["copy_x_to_layer"]

    # Optional gene/cell filtering before spatial neighbours.
    if cfg.get("filter_genes_min_cells"):
        sctm.pp.filter_genes(adata, cfg["filter_genes_min_cells"], layer=layer)
    if cfg.get("filter_cells_min_genes"):
        sq.gr.spatial_neighbors  # no-op guard, filtering below
        import scanpy as sc
        sc.pp.filter_cells(adata, min_genes=cfg["filter_cells_min_genes"])

    # Build spatial neighbours if not present.
    if "spatial_connectivities" not in adata.obsp:
        n_neighs = cfg.get("n_neighs", round(1 / 1000 * adata.n_obs))
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type="generic")

    # Restrict to highly-variable genes if requested (matches the notebooks).
    if cfg.get("use_highly_variable"):
        adata = adata[:, adata.var.highly_variable].copy()

    return adata, layer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", type=Path, default="datasets.json")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=1296)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    cfg = json.load(open(args.config))[args.dataset]
    adata, layer = load_and_prepare(Path(cfg["path"]), cfg)

    n_topics = cfg.get("n_topics", 15)
    out = args.output / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for sw in SPATIAL_WEIGHTS:
        sctm.seed.seed_everything(0)
        model = sctm.stamp.STAMP(
            adata,
            n_topics=n_topics,
            layer=layer,
            gene_likelihood="nb",
            spatial_weight=sw,
            spatial_weight_kwargs={"sigma": cfg.get("gaussian_sigma", 1.0)},
        )
        model.train(
            max_epochs=args.epochs,
            min_epochs=args.epochs,
            device=args.device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        metrics = {k: float(v) for k, v in model.get_metrics(topk=args.topk, layer=layer).items()}
        metrics["Perplexity"] = float(
            model.get_perplexity(device=args.device, dataset="train", num_particles=50)
        )
        metrics["spatial_weight"] = sw
        metrics["n_topics"] = n_topics
        summary_rows.append(metrics)

        run_dir = out / sw
        run_dir.mkdir(parents=True, exist_ok=True)
        model.get_cell_by_topic().to_csv(run_dir / "cell_topic.csv")
        model.get_feature_by_topic().to_csv(run_dir / "gene_topic.csv")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"[{args.dataset}/{sw}] done: {json.dumps(metrics)}")

    # Write combined summary.
    import pandas as pd
    summary = pd.DataFrame(summary_rows)
    summary_path = out / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
