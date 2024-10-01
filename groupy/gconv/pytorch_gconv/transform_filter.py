import torch
import einops
# import tensorflow as tf
from groupy.gconv.make_gconv_indices import *

def transform_filter_3d_nhwc(w, flat_indices, shape_info, validate_indices=True):
    no, nto, ni, nti, n = shape_info
    w_flat = torch.reshape(w, (n * n * n * nti, ni, no))  # shape (n * n * n * nti, ni, no)

    # Adjust the flat_indices to match the shape of w_flat for gather operation
    expanded_indices = flat_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, ni, no)  # shape (nto, nti, ni, no)

    # Do the transformation / indexing operation.
    transformed_w = torch.gather(w_flat, 0, expanded_indices)  # shape (nto, nti, ni, no)

    # Rearrange axes using einops and reshape to get a standard shape filter bank
    transformed_w = einops.rearrange(transformed_w, 'nto nti ni no -> (ni nti) (no nto)')

    return transformed_w

def flatten_indices(inds):
    # Assuming inds is a tensor of shape (nto, nti, n, n)
    # This function should be implemented to flatten the indices as per your specific needs
    flat_inds = inds.view(-1)
    return flat_inds

def torch_trans_filter(w, inds):
    flat_inds = flatten_indices_3d(inds)
    no, ni, nti, n, _ = w.shape
    shape_info = (no, inds.shape[0], ni, nti, n)

    w = w.permute(3, 4, 2, 1, 0).reshape((n, n, nti * ni, no))  # Transpose and reshape

    wt = torch.tensor(w)
    rwt = transform_filter_3d_nhwc(wt, flat_inds, shape_info)

    nto = inds.shape[0]
    rwt = rwt.permute(3, 2, 0, 1).reshape(no, nto, ni, nti, n, n)  # Transpose and reshape
    return rwt


if __name__ == "__main__":
    w = 