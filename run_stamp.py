"""Train upstream STAMP on a local h5ad file; no automatic data downloads."""
import argparse
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
args = parser.parse_args()
if not args.input.is_file():
    parser.error('Input h5ad does not exist; supply your own data file.')
adata = ad.read_h5ad(args.input)
if args.layer not in adata.layers:
    parser.error('Missing counts layer: '+args.layer)
if 'spatial' not in adata.obsm:
    parser.error("Missing adata.obsm['spatial']")
sctm.seed.seed_everything(0)
if 'spatial_connectivities' not in adata.obsp:
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=8)
model = sctm.stamp.STAMP(adata, n_topics=args.topics, layer=args.layer)
model.train(max_epochs=args.epochs, min_epochs=args.epochs, device=args.device)
args.output.mkdir(parents=True, exist_ok=True)
model.get_cell_by_topic().to_csv(args.output/'cell_topic.csv')
model.get_feature_by_topic().to_csv(args.output/'gene_topic.csv')
model.save(str(args.output/'pyro_params.pt'))
print('Results saved to', args.output)
