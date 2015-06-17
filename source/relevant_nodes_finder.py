import numpy as np

from scipy.sparse.dok import dok_matrix
from math import ceil, floor


class RelevantNodesFinder:
    """
    Applies bi-directional search to find the nodes in the
    shortest paths between every term-pair in the dataset.
    """

    def __init__(self, adjacency_matrix):
        """Init the relevant nodes search
        :param adjacency_matrix -- the resource adjacency matrix
        """
        
        self.adjacency_matrix = adjacency_matrix
        self.transposed_adjacency_matrix = adjacency_matrix.T

    def find_relevant_nodes(self, x, y, max_length): 
        """Finds all nodes in the paths between x and y subject to the maximum 
        length, using matrix operations. 
        :param x -- the ID of the first term
        :param y -- the ID of the second term
        :param max_length -- the maximum path length
        """
    
        m = self.adjacency_matrix
        mT = self.transposed_adjacency_matrix
        
        # Get the dimension (number of nodes)
        dim = m.shape[0]        
        
        n_r = create_one_hot_vector(x, dim)
        n_g = create_one_hot_vector(y, dim)
        
        return find_nodes(m, mT, n_r, n_g, max_length)


def find_nodes(m, mT, n_r, n_g, max_len):
    """Finds all nodes in the paths between x and y subject to the maximum 
    length (len). 
    :param m -- the resource adjacency matrix
    :param mT -- the transposed adjacency matrix
    :param n_r -- the one-hot vector representing the root node
    :param n_g -- the one-hot vector representing the goal node
    :param max_len -- the maximum path length
    """

    nodes = set()
    n_x = n_r
    n_y = n_g

    # Stop condition 1 - no paths
    if max_len == 0:
        return nodes

    # Stop condition 2 - the two sides are connected by one edge.
    # Notice that if max_length == 1, then this function will return the two
    # nodes even if they are not connected - this path will be discarded
    # in the second search phase.
    if max_len == 1:
        return set(n_r.nonzero()[1].flatten()).union(set(n_g.nonzero()[1].flatten()))
    
    # Move one step in each direction until the root and goal meet
    for l in range(max_len + 1):
        
        # The root and goal met - apply recursively for each half of the path
        if n_r.dot(n_g.T)[0, 0] > 0:
            
            intersection = n_r.multiply(n_g)           
            forward = find_nodes(m, mT, n_x, intersection, int(ceil((l + 1) / 2.0)))
            backward = find_nodes(m, mT, intersection, n_y, int(floor((l + 1) / 2.0)))
            return forward.union(backward)
        
        # Make a step forward
        if l % 2 == 0:
            n_r = n_r.dot(m)
        # Make a step backward        
        else:
            n_g = n_g.dot(mT)
    
    return nodes
        

def create_one_hot_vector(x, dim):
    """Creates the one-hot vector representing this node
    :param x -- the node
    :param dim -- the number of nodes (the adjacency matrix dimension)
    """
    
    n_x = dok_matrix((1, dim), dtype=np.int16)
    n_x[0, x] = 1
    n_x = n_x.tocsr()
    return n_x