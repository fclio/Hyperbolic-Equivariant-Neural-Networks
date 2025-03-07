
# Code for generating indices used in G-convolutions for various groups G.
# The indices created by these functions are used to rotate and flip filters on the plane or on a group.
# These indices depend only on the filter size, so they are created only once at the beginning of training.

import numpy as np

from groupy.garray.C4_array import C4
from groupy.garray.D4_array import D4
from groupy.garray.p4_array import C4_halfshift
from groupy.gfunc.z2func_array import Z2FuncArray
from groupy.gfunc.p4func_array import P4FuncArray
from groupy.gfunc.p4mfunc_array import P4MFuncArray

# Creates indices for a convolution filter under C4 symmetry group 
# (4 rotations: 0°, 90°, 180°, 270°).
def make_c4_z2_indices(ksize):
#    This initializes a random convolutional filter of shape (1, ksize, ksize)
    x = np.random.randn(1, ksize, ksize)  # A random filter (just for illustration)
    f = Z2FuncArray(v=x)  # Wraps the filter x into a Z2FuncArray (group element for Z2 group)

    if ksize % 2 == 0:
        # Uses left_translation_indices to compute the new positions
        # of each filter weight after rotation.
        uv = f.left_translation_indices(C4_halfshift[:, None, None, None])
        
    else:
        # For odd-sized filters, use C4 group indices
        uv = f.left_translation_indices(C4[:, None, None, None])
  

    # Create a Placeholder for Indices
    # r is initialized as an array of zeros with the same shape as uv, but with an extra dimension.
    r = np.zeros(uv.shape[:-1] + (1,))  # Create an empty array (shape adjusted for further manipulation)

    # Combine Rotation Index and Translations
    ruv = np.c_[r, uv] 

    return ruv.astype('int32')  # Return the final indices array


def make_c4_p4_indices(ksize):
    x = np.random.randn(4, ksize, ksize)
    f = P4FuncArray(v=x)

    if ksize % 2 == 0:
        li = f.left_translation_indices(C4_halfshift[:, None, None, None])
    else:
        li = f.left_translation_indices(C4[:, None, None, None])
    return li.astype('int32')

#  you start with a single filter, 
#  and then you apply multiple symmetry transformations to that single filter. 
#  The extra dimension ((8, 1, ksize, ksize, 1)) helps manage the multiple transformations 
#  and adds a "marker" to hold transformation indices.
def make_d4_z2_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(1, ksize, ksize)
    f = Z2FuncArray(v=x)
    # D4.flatten() is an array representing all possible symmetry operations in the D4 group. It is likely a set of transformations that include both rotations and reflections (e.g., [0°, 90°, 180°, 270°, horizontal flip, vertical flip, diagonal flip, etc.]
    uv = f.left_translation_indices(D4.flatten()[:, None, None, None])
    #  computes the indices (coordinates) of how the kernel elements move under the D4 symmetry transformations.
    # This operation applies these transformations to the filter f and stores the new positions for each symmetry.
 
    # (8, 1, 3, 3, 2)
    # (8 transformation, 1 input, ksize, ksize, coordinate_position(u,v))
    
    # The new array mr is initialized as a zero array with the same shape as uv, except the last dimension.
    mr = np.zeros(uv.shape[:-1] + (1,))
    # (8, 1, 3, 3, 1)
    mruv = np.c_[mr, uv]
    # (8, 1, 3, 3, 3)
    # (8 transformation, 1 input, ksize, ksize, coordinate_position(group_index,u,v))
    
    return mruv.astype('int32')

# you already have 8 separate filters,
#  each undergoing the same set of transformations
# . There's no need to add an extra dimension because the transformations are applied independently to each of the 8 filters.
def make_d4_p4m_indices(ksize):
    assert ksize % 2 == 1  # TODO
    x = np.random.randn(8, ksize, ksize)
    f = P4MFuncArray(v=x)
    li = f.left_translation_indices(D4.flatten()[:, None, None, None])
    # (8, 8, 3, 3, 3)
    # (8 transformation, 1 input, ksize, ksize, coordinate_position(group_index,u,v))
    
    return li.astype('int32')


def flatten_indices(inds):
    """
    The Chainer implementation of G-Conv uses indices into a 5D filter tensor (with an additional axis for the
    transformations H. For the tensorflow implementation it was more convenient to flatten the filter tensor into
    a 3D tensor with shape (output channels, input channels, transformations * width * height).

    This function takes indices in the format required for Chainer and turns them into indices into the flat array
    used by tensorflow.

    :param inds: np.ndarray of shape (output transformations, input transformations, n, n, 3), as output by
    the functions like make_d4_p4m_indices(n).
    :return: np.ndarray of shape (output transformations, input transformations, n, n)
    """
    n = inds.shape[-2]
    nti = inds.shape[1]
    T = inds[..., 0]  # shape (nto, nti, n, n)
    U = inds[..., 1]  # shape (nto, nti, n, n)
    V = inds[..., 2]  # shape (nto, nti, n, n)
    # inds_flat = T * n * n + U * n + V
    inds_flat = U * n * nti + V * nti + T
    return inds_flat