import numpy as np
import codecs

from scipy.sparse.dok import dok_matrix
from matrix_serializer import save_matrix, load_matrix
from logger import initialize_logger, log_info


class KnowledgeResource:
    """
    Holds the resource graph data
    """

    def __init__(self, resource_mat_file, entity_map_file, property_map_file, \
                 relations_file, allow_reversed_edges, relevant_nodes=None):
        """Init the knowledge resource
        :param resource_mat_file -- the resource adjacency matrix file
        :param entity_map_file -- the resource term to node id map file
        :param property_map_file -- the resource property label to id map file
        :param relations_file -- the resource edges file
        :param allow_reversed_edges -- whether reversed edges are allowed in this resource
        """

        self.allow_reversed_edges = allow_reversed_edges

        self.adjacency_matrix = load_resource_matrix(resource_mat_file, allow_reversed_edges)
            
        self.term_to_id, self.id_to_term = load_map(entity_map_file, relevant_nodes)
        self.prop_to_id, self.id_to_prop = load_map(property_map_file, None) 
        self.l2r_edges, self.r2l_edges = load_edges(relations_file, relevant_nodes)

    def get_nodes_edges(self, nodes):
        """Returns the edges e=(x,y) such that x,y in nodes"""
        l2r_edges = self.l2r_edges
        r2l_edges = self.r2l_edges
        reduced_l2r = {}
        reduced_r2l = {}

        lists = [(l2r_edges, reduced_l2r), (r2l_edges, reduced_r2l)]
        for (orig, new) in lists:
            for x in orig:
                if x in nodes:
                    for y in orig[x]:
                        if y in nodes:
                            if x not in new:
                                new[x] = {}
                            new[x][y] = orig[x][y]

        return reduced_l2r, reduced_r2l


def load_resource_matrix(resource_mat_file, allow_reversed_edges):
    """Loads the resource matrix
    :param resource_mat_file -- the resource adjacency matrix file
    :param allow_reversed_edges -- whether reversed edges are allowed in this resource
    """
        
    # Read the matrix from the file to compressed format
    compressed_file = resource_mat_file
    
    if not resource_mat_file.endswith('.npz'):
        adjacency_matrix = read_matrix(resource_mat_file)
        log_info('Read matrix.')
        save_matrix(resource_mat_file + '.tmp', adjacency_matrix)
        log_info('Saved matrix.')
        compressed_file = compressed_file + '.tmp'
        
    adjacency_matrix = load_matrix(compressed_file)
    log_info('Loaded matrix.')
    
    # Add the transposed matrix, since we are looking for paths
    # in both directions
    if allow_reversed_edges:
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        log_info('Added transpose.')
        
    return adjacency_matrix
    
        
def read_matrix(path):
    """Read a MatrixMarket format file to a matrix object"""
    
    with open(path) as f:
        first = True
        second = False
        for line in f:
            if first:
                first = False
                second = True
                continue
            elif second:
                tokens = line.strip().split()
                dim_x, dim_y = int(tokens[0]), int(tokens[1])
                m = dok_matrix((dim_x, dim_y), dtype=np.int16)
                second = False
                continue
            x, y, v = [int(t) for t in line.strip().split()]
            m[x-1, y-1] = v
    return m.tocsr()

    
def load_map(map_file, relevant_nodes):
    """Loads the map of term/property string to ID
    :param map_file -- the map file
    :param relevant_nodes -- the nodes to load from this resource
    """

    str_to_id = {}
    id_to_str = {}
    with codecs.open(map_file, encoding='utf-8') as f:
        for line in f:
            term_id, term_str = line.strip().split('\t')
            term_id = int(term_id)
            
            # Node in relevant nodes or all nodes are included
            if (relevant_nodes is None) or (term_id in relevant_nodes):
                str_to_id[term_str] = term_id
                id_to_str[term_id] = term_str
    return str_to_id, id_to_str


def load_edges(relations_file, relevant_nodes):
    """Loads the left-to-right and right-to-left edges from a file
    :param relations_file -- the left-to-right edges file
    :param relevant_nodes -- the nodes to load from this resource
    """
    l2r_edges = {}
    r2l_edges = {}
    
    with open(relations_file) as f:
        for line in f:
            x, y, prop = map(int, line.strip().split('\t'))
            
            # Node in relevant nodes or all nodes are included
            if (relevant_nodes is None) or ((x in relevant_nodes) and (y in relevant_nodes)):
                if x not in l2r_edges:
                    l2r_edges[x] = {}       
                    
                if y not in l2r_edges[x]:
                    l2r_edges[x][y] = set()
                
                l2r_edges[x][y].add(prop)
                
                if y not in r2l_edges:
                    r2l_edges[y] = {}      
                    
                if x not in r2l_edges[y]:
                    r2l_edges[y][x] = set()
                    
                r2l_edges[y][x].add(prop)

    return l2r_edges, r2l_edges