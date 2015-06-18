import sys
import codecs

from logger import initialize_logger, log_info
from relevant_nodes_finder import RelevantNodesFinder
from knowledge_resource import KnowledgeResource
from path_finder import PathFinder
from docopt import docopt


def main():
    """
    Applies bi-directional search in two phases to find the shortest paths
    between every term-pair in the dataset: the first phase finds the nodes
    along the shortest paths, while the second reconstructs the paths themselves.
    """

    # Get the arguments
    args = docopt("""Find the shortest paths between every term-pair in the dataset.

    Usage:
        search.py <dataset_path> <resource_matrix_path>
        <resource_entities_path> <resource_properties_path>
        <resource_l2r_path> <max_path_length>
        <allow_reversed_edges> <find_relevant_nodes>
        <relevant_nodes_file> <paths_out_file>

        <dataset_path> = the dataset file
        <resource_matrix_path> = the resource adjacency matrix file (.mm/.npz)
        <resource_entities_path> = the entity str-id map file
        <resource_properties_path> = the property str-id map file
        <resource_l2r_path> = the edges file
        <max_path_length> = the maximum path length
        <allow_reversed_edges> = whether reversed edges are allowed in this resource
        <find_relevant_nodes> = whether to find the relevant nodes (or use the results file)
        <relevant_nodes_file> = relevant nodes file (input / output)
        <paths_out_file> = the paths file (output)
    """)

    dataset_file = args['<dataset_path>']
    resource_mat_file = args['<resource_matrix_path>']
    entity_map_file = args['<resource_entities_path>']
    property_map_file = args['<resource_properties_path>']
    edges_file = args['<resource_l2r_path>']
    max_length = int(args['<max_path_length>'])
    allow_reversed_edges = args['<allow_reversed_edges>'][0].upper() == 'T'
    do_find_relevant_nodes = args['<find_relevant_nodes>'][0].upper() == 'T'
    relevant_nodes_file = args['<relevant_nodes_file>']
    paths_file = args['<paths_out_file>']

    initialize_logger()

    # Find relevant nodes  
    if do_find_relevant_nodes:

        # Load the resource
        resource = KnowledgeResource(resource_mat_file, entity_map_file,
                                     property_map_file, edges_file, allow_reversed_edges)
        adjacency_matrix = resource.adjacency_matrix
        term_to_id = resource.term_to_id

        # Load the dataset
        dataset = load_data_labels(dataset_file, adjacency_matrix)

        node_finder = RelevantNodesFinder(adjacency_matrix)
        relevant_nodes = find_relevant_nodes(dataset, max_length, relevant_nodes_file, term_to_id, node_finder)
    else:

        # Load the resource partially according to the relevant nodes
        relevant_nodes = load_relevant_nodes(relevant_nodes_file)

        resource = KnowledgeResource(resource_mat_file, entity_map_file,
                                     property_map_file, edges_file, allow_reversed_edges, get_all_nodes(relevant_nodes))
        adjacency_matrix = resource.adjacency_matrix
        term_to_id = resource.term_to_id

        # Load the dataset
        dataset = load_data_labels(dataset_file, adjacency_matrix)

    path_finder = PathFinder(resource)
    paths_output = open(paths_file, 'w')

    pair_num = 0

    # For each term-pair, find relevant nodes and then find paths
    for (x, y) in dataset.keys():

        pair_num = pair_num + 1

        x_id = -1
        if x in term_to_id:
            x_id = term_to_id[x]

        y_id = -1
        if y in term_to_id:
            y_id = term_to_id[y]

        # Limit the search space using the relevant nodes and find paths
        nodes = relevant_nodes[(x_id, y_id)]
        l2r_edges, r2l_edges = resource.get_nodes_edges(nodes)
        paths = path_finder.find_shortest_paths(x_id, y_id, max_length, l2r_edges, r2l_edges)
        paths_output.write('pair number ' + str(pair_num) + ': ' + x + '->' + y + '\n')
        for path in paths:
            paths_output.write(nice_print_path(path, resource.id_to_prop) + '\n')

    paths_output.close()


def load_data_labels(data_labels_file, resource_mat_file):
    """Loads the dataset file"""
    data_set = {}
    with codecs.open(data_labels_file, encoding='utf-8') as f:
        for line in f:
            x, y, c = line.strip().split('\t')
            data_set[(x, y)] = int(c[0] == 'T')
    return data_set


def load_relevant_nodes(relevant_nodes_file):
    """Loads the results of the find_relevant_nodes phase"""
    relevant_nodes = {}
    with open(relevant_nodes_file) as f:
        for line in f:
            x, y, nodes = line.strip().split('\t')
            nodes = nodes[1:-1].split(', ')
            if '' in nodes:
                nodes.remove('')
            relevant_nodes[(int(x), int(y))] = map(int, nodes)
    return relevant_nodes


def find_relevant_nodes(dataset, max_length, nodes_file, term_to_id, node_finder):
    """For each term-pair find the nodes in the paths between them,
    subject to the maximum length max_length
    :param dataset -- the annotated term-pairs
    :param max_length -- the maximum path length
    :param nodes_file -- relevant nodes file (output)
    :param term_to_id -- the map of term string and int ID
    :param node_finder -- a NodeFinder object
    """
    relevant_nodes = {}
    with open(nodes_file, 'w') as nodes_output:
        for (x, y) in dataset.keys():

            x_id = -1
            if x in term_to_id:
                x_id = term_to_id[x]

            y_id = -1
            if y in term_to_id:
                y_id = term_to_id[y]

            # Find the relevant nodes
            nodes = node_finder.find_relevant_nodes(x_id, y_id, max_length)
            nodes_output.write('\t'.join([str(x_id), str(y_id), str(list(nodes))]) + '\n')
            relevant_nodes[(x_id, y_id)] = nodes

    return relevant_nodes


def nice_print_path(path, id_to_prop):
    """Receives a sequence of edge types IDs and returns
    the string representation of the path type"""
    path_str = '^'

    for edge_type in path:
        label = int(edge_type[0])
        direction = edge_type[1]

        if direction == PathFinder.R2L:
            path_str = path_str + '<'
        path_str = path_str + '-' + id_to_prop[label] + '-'

        if direction == PathFinder.L2R:
            path_str = path_str + '>'
        path_str = path_str + ' '

    path_str = path_str[:-1] + '$'
    return path_str


def get_all_nodes(relevant_nodes):
    """Return all nodes along any path in this dataset"""
    nodes = set()

    for (x, y) in relevant_nodes:
        nodes.add(x)
        nodes.add(y)

        for node in relevant_nodes[(x, y)]:
            nodes.add(node)

    return nodes


if __name__ == '__main__':
    main()
