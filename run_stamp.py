"""Train upstream STAMP on a local h5ad file; no automatic data downloads."""
import argparse
import json
from pathlib import Path

import anndata as ad
import squidpy as sq
import sctm

parser = argparse.ArgumentParser()
parser.add_argument('input', type=Path)
parser.add_argument('--topics', type=int, default=5)
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--layer', default='counts')
parser.add_argument('--device', default='cpu')
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'outputs'/'stamp')
parser.add_argument('--config', type=Path, default=None,
                    help='Optional JSON config file (see config.json). CLI args override it.')
parser.add_argument('--spatial-weight', default=None,
                    choices=['binary', 'gaussian', 'cosine', 'combined'],
                    help='Override spatial_weight.method from the config.')
parser.add_argument('--use-amp', action='store_true', default=None,
                    help='Enable autocast mixed precision (overrides config).')
parser.add_argument('--sparse-storage', action='store_true', default=None,
                    help='Keep counts as sparse tensor (overrides config).')
parser.add_argument('--top-genes', type=int, default=50,
                    help='Top genes per topic in the exported viz modules.')
args = parser.parse_args()

if not args.input.is_file():
    parser.error('Input h5ad does not exist; supply your own data file.')

# --- Load config (all keys optional, defaults = upstream behaviour) ---------
cfg = {"spatial_weight": {}, "engineering": {}}
if args.config is not None:
    if not args.config.is_file():
        parser.error('Config file does not exist: ' + str(args.config))
    with open(args.config, encoding='utf-8') as f:
        cfg = json.load(f)
    cfg.setdefault("spatial_weight", {})
    cfg.setdefault("engineering", {})

spatial_weight = cfg["spatial_weight"]
method = args.spatial_weight or spatial_weight.get("method", "binary")
sw_kwargs = {
    "sigma": spatial_weight.get("gaussian_sigma", 1.0),
    "layer": spatial_weight.get("cosine_layer", None),
    "alpha": spatial_weight.get("combined_alpha", 0.5),
}
use_amp = cfg["engineering"].get("use_amp", False) if args.use_amp is None else args.use_amp
sparse_storage = (
    cfg["engineering"].get("sparse_storage", False)
    if args.sparse_storage is None
    else args.sparse_storage
)

adata = ad.read_h5ad(args.input)
if args.layer not in adata.layers:
    parser.error('Missing counts layer: '+args.layer)
if 'spatial' not in adata.obsm:
    parser.error("Missing adata.obsm['spatial']")
sctm.seed.seed_everything(0)
if 'spatial_connectivities' not in adata.obsp:
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=8)

model = sctm.stamp.STAMP(
    adata,
    n_topics=args.topics,
    layer=args.layer,
    spatial_weight=method,
    spatial_weight_kwargs=sw_kwargs,
    use_amp=use_amp,
    sparse_storage=sparse_storage,
)
model.train(max_epochs=args.epochs, min_epochs=args.epochs, device=args.device)

args.output.mkdir(parents=True, exist_ok=True)
model.get_cell_by_topic().to_csv(args.output/'cell_topic.csv')
model.get_feature_by_topic().to_csv(args.output/'gene_topic.csv')
model.save(str(args.output/'pyro_params.pt'))
viz_path = sctm.export.export_viz_data(model, args.output, top_genes=args.top_genes, layer=args.layer)
print('Results saved to', args.output, '(viz:', viz_path.name + ')')
