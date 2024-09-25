import numpy as np
import torch
import pdb

# from numba import jit, njit


# @jit
def jaccard_overlap(localizations_a, localizations_b):
    """Jaccard overlap between two segments A ∩ B / (LENGTH_A + LENGTH_B - A ∩ B)

    localizations_a: tensor of localizations
    localizations_a: tensor of localizations
    """
    A = localizations_a.shape[0]
    B = localizations_b.shape[0]
    max_min = np.maximum(
        np.tile(localizations_a[:, 0][:, None], [1, B]), np.tile(localizations_b[:, 0][None, :], [A, 1])
    )
    min_max = np.minimum(
        np.tile(localizations_a[:, 1][:, None], [1, B]), np.tile(localizations_b[:, 1][None, :], [A, 1])
    )
    intersection = np.clip(min_max - max_min, a_min=0, a_max=None)
    length_a = np.tile((localizations_a[:, 1] - localizations_a[:, 0])[:, None], [1, B])
    length_b = np.tile((localizations_b[:, 1] - localizations_b[:, 0])[None, :], [A, 1])
    if (length_a + length_b - intersection == 0).any():
        pdb.set_trace()
    try:
        overlaps = intersection / (length_a + length_b - intersection)
    except RuntimeWarning:
        pdb.set_trace()
    # max_min = torch.max(localizations_a[:, 0].unsqueeze(1).expand(A, B),
    #                     localizations_b[:, 0].unsqueeze(0).expand(A, B))

    # min_max = torch.min(localizations_a[:, 1].unsqueeze(1).expand(A, B),
    #                     localizations_b[:, 1].unsqueeze(0).expand(A, B))
    # intersection = torch.clamp((min_max - max_min), min=0)
    # length_a = (localizations_a[:, 1] - localizations_a[:, 0]).unsqueeze(1).expand(A, B)
    # length_b = (localizations_b[:, 1] - localizations_b[:, 0]).unsqueeze(0).expand(A, B)
    # overlaps = intersection / (length_a + length_b - intersection)
    return overlaps


# def jaccard_overlap(localizations_a, localizations_b):
#     """Jaccard overlap between two segments A ∩ B / (LENGTH_A + LENGTH_B - A ∩ B)

#     localizations_a: tensor of localizations
#     localizations_a: tensor of localizations
#     """
# A = localizations_a.size(0)
# B = localizations_b.size(0)
# max_min = torch.max(localizations_a[:, 0].unsqueeze(1).expand(A, B),
#                     localizations_b[:, 0].unsqueeze(0).expand(A, B))

# min_max = torch.min(localizations_a[:, 1].unsqueeze(1).expand(A, B),
#                     localizations_b[:, 1].unsqueeze(0).expand(A, B))
# intersection = torch.clamp((min_max - max_min), min=0)
# length_a = (localizations_a[:, 1] - localizations_a[:, 0]).unsqueeze(1).expand(A, B)
# length_b = (localizations_b[:, 1] - localizations_b[:, 0]).unsqueeze(0).expand(A, B)
# overlaps = intersection / (length_a + length_b - intersection)
#     return overlaps
