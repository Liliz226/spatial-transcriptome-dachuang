"""Run upstream STAMP on synthetic counts, without downloading data."""
from pathlib import Path
import json
import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.neighbors import kneighbors_graph
import sctm

sctm.seed.seed_everything(0)
rng = np.random.default_rng(0)
xy = np.array([(i, j) for i in range(8) for j in range(8)])
rates = np.ones((64, 32), dtype=np.float32)
rates[xy[:, 0] < 4, :16] = 6
rates[xy[:, 0] >= 4, 16:] = 6
adata = ad.AnnData(sparse.csr_matrix(rng.poisson(rates).astype(np.float32)))
adata.layers['counts'] = adata.X.copy()
adata.obsm['spatial'] = xy
adj = kneighbors_graph(xy, 4, mode='connectivity', include_self=False)
adata.obsp['spatial_connectivities'] = adj.maximum(adj.T).tocsr()
model = sctm.stamp.STAMP(adata, n_topics=2, hidden_size=16, layer='counts')
model.train(max_epochs=3, min_epochs=3, batch_size=32, device='cpu', early_stop=False)
theta = model.get_cell_by_topic()
beta = model.get_feature_by_topic()
assert theta.shape == (64, 2), theta.shape
assert beta.shape == (32, 2), beta.shape
assert np.isfinite(theta.to_numpy()).all()
assert np.isfinite(beta.to_numpy()).all()
assert np.allclose(theta.sum(axis=1), 1, atol=1e-4)
out = Path(__file__).resolve().parent/'outputs'/'smoke'
out.mkdir(parents=True, exist_ok=True)
theta.to_csv(out/'cell_topic.csv')
beta.to_csv(out/'gene_topic.csv')
(out/'result.json').write_text(json.dumps({'status':'passed','cells':64,'genes':32,'topics':2,'epochs':3,'device':'cpu'},indent=2))
print('PASS: synthetic STAMP training and topic extraction', out)
