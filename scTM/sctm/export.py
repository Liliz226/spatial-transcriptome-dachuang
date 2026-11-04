"""Serialize STAMP outputs for the downstream web visualization layer (创新点二).

The report-layer UI (Vue + ECharts) needs three artefacts: cell-by-topic
proportions, spatial coordinates and per-topic top-gene modules. This module
gathers them into a single ``viz_data.json`` without depending on any front-end
stack, so the training pipeline can emit a ready-to-render payload.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Index,)):
        return list(obj)
    return obj


def export_viz_data(model, out_dir, top_genes=50, layer=None):
    """Write ``viz_data.json`` for the three-layer visualization system.

    Args:
        model: A trained ``sctm.stamp.STAMP`` instance.
        out_dir: Directory (``Path`` or ``str``) to write into.
        top_genes: Number of top genes per topic in the module ranking table.
        layer: Counts layer name (informational only, stored in metadata).

    Returns:
        ``Path`` of the written file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    theta = model.get_cell_by_topic()
    beta = model.get_feature_by_topic()

    adata = model.adata
    coords = (
        adata.obsm["spatial"] if "spatial" in adata.obsm else np.empty((model.n_obs, 0))
    )

    modules = {}
    for topic in beta.columns:
        top = beta.nlargest(top_genes, topic)
        modules[topic] = [
            {"gene": str(g), "weight": float(top.loc[g, topic])} for g in top.index
        ]

    payload = {
        "meta": {
            "n_cells": int(model.n_obs),
            "n_genes": int(model.n_features),
            "n_topics": int(model.n_topics),
            "layer": layer,
            "top_genes": int(top_genes),
        },
        "cells": {
            "names": list(theta.index),
            "coordinates": _jsonable(coords),
            "topic_proportions": {
                col: _jsonable(theta[col].values) for col in theta.columns
            },
        },
        "gene_modules": modules,
    }

    path = out_dir / "viz_data.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_jsonable)
    return path
