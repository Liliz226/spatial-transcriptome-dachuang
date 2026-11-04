"""Dynamic spatial-weight encoders for the adjacency matrix (创新点一).

The upstream STAMP uses a fixed binary ``spatial_connectivities`` graph: every
edge contributes weight 1 and the only signal is ``deg @ adj @ deg`` symmetric
normalization. This module replaces that fixed graph with edge weights that
combine (a) Gaussian spatial-distance decay and (b) expression cosine
similarity, while preserving the same sparse CSR structure so the existing
``precompute_SGC_scipy`` normalizer works unchanged.

All functions return a ``scipy.sparse.csr_matrix`` of float32 weights in
``[0, 1]`` with exactly the same sparsity pattern as the input adjacency.
"""

import numpy as np
from scipy import sparse

from .utils import sparsify

_VALID_METHODS = ("binary", "gaussian", "cosine", "combined")


def _iter_rows(A):
    """Yield ``(row_idx, neighbor_cols)`` for each row of a CSR matrix."""
    for i in range(A.shape[0]):
        cols = A.indices[A.indptr[i] : A.indptr[i + 1]]
        if cols.size:
            yield i, cols


def _gaussian_weights(A, coords, sigma):
    """Distance-decay weights on the existing edge structure."""
    w = np.zeros(A.nnz, dtype="float32")
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for i, cols in _iter_rows(A):
        s = A.indptr[i]
        d = np.linalg.norm(coords[cols] - coords[i], axis=1)
        w[s : A.indptr[i + 1]] = np.exp(-(d * d) * inv2s2)
    return w


def _cosine_weights(A, X):
    """Cosine similarity of expression vectors on the existing edge structure."""
    X = X.tocsr()
    rn = np.asarray(np.sqrt(X.multiply(X).sum(axis=1))).ravel()
    rn[rn == 0] = 1.0
    w = np.zeros(A.nnz, dtype="float32")
    for i, cols in _iter_rows(A):
        s = A.indptr[i]
        xi = X.getrow(i)
        num = np.asarray((xi @ X[cols].T).toarray()).ravel()
        denom = rn[i] * rn[cols]
        w[s : A.indptr[i + 1]] = np.clip(num / np.maximum(denom, 1e-12), 0.0, 1.0)
    return w


def build_spatial_weights(adata, method="binary", layer=None, sigma=1.0, alpha=0.5):
    """Build the (possibly weighted) adjacency matrix.

    Args:
        adata: AnnData with ``obsp['spatial_connectivities']`` and, for
            gaussian, ``obsm['spatial']`` coordinates.
        method: One of ``binary`` (original), ``gaussian``, ``cosine``,
            ``combined``. Default ``binary`` = upstream behaviour.
        layer: Counts layer used for cosine similarity. ``None`` uses ``X``.
        sigma: Gaussian lengthscale for distance decay.
        alpha: Weight of the distance term in ``combined``
            (``alpha * dist + (1 - alpha) * cosine``).

    Returns:
        ``scipy.sparse.csr_matrix`` of float32 weights with the same
        sparsity pattern as ``spatial_connectivities``.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")

    if "spatial_connectivities" not in adata.obsp:
        raise KeyError("spatial_connectivities not found in adata.obsp")

    A = adata.obsp["spatial_connectivities"].astype("float32").tocsr()

    if method == "binary":
        return A

    w_dist = w_cos = None
    if method in ("gaussian", "combined"):
        if "spatial" not in adata.obsm:
            raise KeyError("obsm['spatial'] required for gaussian weights")
        w_dist = _gaussian_weights(A, adata.obsm["spatial"], sigma)

    if method in ("cosine", "combined"):
        w_cos = _cosine_weights(A, sparsify(adata, layer=layer))

    W = A.copy()
    if method == "gaussian":
        W.data = w_dist
    elif method == "cosine":
        W.data = w_cos
    else:  # combined
        W.data = alpha * w_dist + (1.0 - alpha) * w_cos

    W.eliminate_zeros()
    return W
