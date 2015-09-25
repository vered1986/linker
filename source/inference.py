import codecs

from relevant_nodes_finder import RelevantNodesFinder
from knowledge_resource import PartialKnowledgeResource
from path_finder import PathFinder
from search import nice_print_path
from docopt import docopt


def main():
    """
    Receives a list of knowledge resources (each represented by a directory containing all
    the resource files), a dataset of unlabeled term-pairs, and the pre-trained whitelist.
    Returns the prediction for every term pair.
    """

    # Get the arguments
    args = docopt("""Predict the label of every term-pair in the dataset, using the pretrained whitelist.

    Usage:
        inference.py <dataset_path> <output_dir> <whitelist_path> <max_length> <resource_prefix>...

        <dataset_path> = the dataset file
        <output_dir> = the output directory
        <whitelist_path> = the pretrained whitelist
        <max_length> = the maximum path length
        <resource_prefix>... = the directories and file name prefix of the resource, containing the matrix, entities, properties and l2r files
    """)

    dataset_file = args['<dataset_path>']
    whitelist_path = args['<whitelist_path>']
    output_dir = args['<output_dir>']
    max_length = int(args['<max_length>'])

    # Load the dataset
    dataset = load_dataset(dataset_file)

    # Load the whitelist
    whitelist = load_whitelist(whitelist_path)

    # Load all the resources
    resources = []
    node_finders = []
    path_finders = []

    for dir in args['<resource_prefix>']:

        resource = PartialKnowledgeResource(dir + 'Matrix.mm.tmp.npz', dir + 'Entities.txt',
                                            dir + 'Properties.txt', dir + '-l2r.txt', whitelist)
        resources.append(resource)
        adjacency_matrix = resource.adjacency_matrix
        node_finders.append(RelevantNodesFinder(adjacency_matrix))
        path_finders.append(PathFinder(resource))

    with codecs.open(output_dir + '/predictions.txt', 'w', 'utf-8') as f_out:

        # For each term-pair, find relevant nodes and then find paths
        for (x, y) in dataset:

            x = x.decode('utf-8')
            y = y.decode('utf-8')

            label = False
            f_out.write('\t'.join([x, y]))

            # Search paths in each resource, using only the allowed paths
            for i, resource in enumerate(resources):

                x_id, y_id, relevant_nodes = find_relevant_nodes(x, y, max_length, resource.term_to_id, node_finders[i])

                # Limit the search space using the relevant nodes and find paths
                l2r_edges, r2l_edges = resource.get_nodes_edges(relevant_nodes)
                paths = path_finders[i].find_shortest_paths(x_id, y_id, max_length, l2r_edges, r2l_edges)

                for path in paths:
                    str_path = nice_print_path(path, resource.id_to_prop)

                    # Make sure that all the properties are in the whitelist
                    if whitelist.issuperset(set(str_path.split(' '))):
                        label = True
                        f_out.write('\t' + '\t'.join(['True', str_path]) + '\n')
                        break

                if label:
                    break

            if not label:
                f_out.write('\tFalse\n')


def load_dataset(dataset_file):
    """Loads the dataset file"""
    data_set = []
    with codecs.open(dataset_file, encoding='utf-8') as f_in:
        data_set = [tuple(line.strip().split('\t')) for line in f_in]
    return data_set


def find_relevant_nodes(x, y, max_length, term_to_id, node_finder):
    """Find the nodes in the paths between x and y,
    subject to the maximum length max_length
    :param x
    :param y
    :param max_length -- the maximum path length
    :param term_to_id -- the map of term string and int ID
    :param node_finder -- a NodeFinder object
    """
    x_id = -1
    if x in term_to_id:
        x_id = term_to_id[x]

    y_id = -1
    if y in term_to_id:
        y_id = term_to_id[y]

    # Find the relevant nodes
    return x_id, y_id, node_finder.find_relevant_nodes(x_id, y_id, max_length)


def load_whitelist(file_path):
    """Loads a predefined whitelist
    :param file_path The whitelist's file
    """
    with open(file_path) as in_file:
        whitelist = [line.strip() for line in in_file]

    return set(whitelist)


if __name__ == '__main__':
    main()
