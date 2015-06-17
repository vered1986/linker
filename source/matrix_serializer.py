import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.dok import dok_matrix

def save_matrix(file, m):
    """Save a matrix into a single file in compressed .npz format
    :param file The file path
    :param m The matrix"""
    np.savez_compressed(file, data=m.data, indices=m.indices, indptr=m.indptr, shape=m.shape)


def load_matrix(file):
    """ Loads a matrix from a compressed file
    :param file: The file Path
    :return: A csr_matrix object
    """
    if not file.endswith('.npz'):
        file += '.npz'
    loader = np.load(file)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def read_mm_matrix(file):
    """Read a MatrixMarket format file to a matrix object
    :param file: The file path
    """
    with open(file) as f:
        first = True
        second = False
        for line in f:

            # Skip the first line
            if first:
                first = False
                second = True
                continue

            # The header is in the second line
            elif second:
                tokens = line.strip().split()
                dim_x, dim_y = int(tokens[0]), int(tokens[1])
                m = dok_matrix((dim_x, dim_y), dtype=np.int16)
                second = False
                continue

            # The rest of the lines are the data
            x, y, v = [int(t) for t in line.strip().split()]
            m[x-1, y-1] = v
    return m.tocsr()