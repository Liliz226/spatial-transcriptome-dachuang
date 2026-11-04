import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset


class DictDataset(Dataset):
    """Dictionary based PyTorch dataset."""

    def __init__(self, tensor_dict, idx_key="sample_idx"):
        self.tensor_dict = tensor_dict
        self.idx_key = idx_key
        # just stores them
        self._len = tensor_dict[next(iter(tensor_dict))].shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        item = {self.idx_key: index}
        scalar = np.isscalar(index)
        for k, v in self.tensor_dict.items():
            val = v[index]
            # Sparse storage mode keeps `x` as a scipy CSR matrix (or torch
            # sparse tensor). Array subsets stay sparse so the train/test split
            # does not materialize the full dense matrix; single rows (scalar
            # index, i.e. the per-batch DataLoader path) are densified here so
            # every downstream consumer receives a dense batch.
            if sparse.issparse(val):
                if scalar:
                    val = torch.from_numpy(val.toarray().ravel())
            elif torch.is_tensor(val) and val.is_sparse:
                if scalar:
                    val = val.to_dense()
            item[k] = val
        return item
