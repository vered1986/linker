import codecs
import re

import numpy as np

from collections import Counter

def load_data(data_dir, orig_schema=None):
    """Loads the dataset (labels) and path file
    :param data_dir -- the data directory
    :param orig_schema -- the edge types list (orig_schema=None
    in the training set, and will be populated with the edge types
    that occur in the training paths).
    """
    data_labels = load_data_labels(data_dir + '_label')
    return load_pairs(data_dir + '_path_short', data_labels, orig_schema)


def load_data_labels(data_labels_file):
    """Loads the dataset file
    :param data_labels -- the dataset (labels) file
    """
    data_set = {}
    with codecs.open(data_labels_file, encoding='utf-8') as f:
        for line in f:
            x, y, c = line.strip().split('\t')
            data_set[(x, y)] = int(c[0] == 'T')
    return data_set


def load_pairs(paths_file, data_labels, orig_schema=None):
    """Load the paths for the pairs in the dataset
    :param paths_file -- the paths file
    :param data_labels -- the dataset (labels) file
    :param orig_schema -- the edge types list (orig_schema=None
    in the training set, and will be populated with the edge types
    that occur in the training paths).
    """
    path_sets = []
    labels = []
    pair_pattern = re.compile('pair number [0-9]+: (.+)->(.+)')

    with codecs.open(paths_file, 'r', 'utf-8') as f:
        for line in f:
            match = pair_pattern.search(line.strip())

            # Pair line
            if match is not None:
                curr_xy = (match.group(1), match.group(2))
                path_sets.append([])
                labels.append(data_labels[curr_xy])

            # Path line
            else:
                path = Counter(line.strip().split())
                path_sets[-1].append(path)

    schema, path_sets = vectorize_paths(path_sets, orig_schema)
    return schema, path_sets, np.array(labels)


def load_raw_pairs(data_dir):
    """Load the paths for the pairs in the dataset
    :param data_dir -- the data directory
    """
    pairs = []
    pair_pattern = re.compile('pair number [0-9]+: (.+)->(.+)')
    
    with open(data_dir + '_path_short') as f:
        for line in f:
            match = pair_pattern.search(line.strip())

            # Pair line
            if match is not None:
                pairs.append((match.group(1), match.group(2)))
    
    return pairs


def vectorize_paths(path_sets, orig_schema=None):
    """Creates pair representation as a set of paths,
    where each path is a bag of edges.
    path_sets[i,j] = the frequency of edge type j in path i.
    :param path_sets -- a list of pairs, each represented by a set of
    paths, each path is a counter of string edge types
    :param orig_schema -- the edge types list (orig_schema=None
    in the training set, and will be populated with the edge types
    that occur in the training paths).
    """

    # Create the schema according to the edge types in the training set
    if orig_schema is None:
        schema = set.union(*[set(path.keys()) for path_set in path_sets for path in path_set])
        schema.add('$')
        schema = sorted(schema)
    else:
        schema = orig_schema

    # the number for each string edge type
    index = dict([(x, i) for i, x in enumerate(schema)])

    vectorized_paths = [[vectorized_path for vectorized_path in [vectorize_path(path, index) for path in path_set]
                         if vectorized_path is not None] for path_set in path_sets]
    return schema, vectorized_paths


def vectorize_path(path, index):
    """Receives a counter of string edge types and returns
    a vector representing the path (bag of edges).
    :param path -- a counter of string edge types
    :param index -- the number for each string edge type
    """
    path_vector = np.zeros(len(index))
    for edge, count in path.items():

        # Path contains edge which is not found in the
        # original schema - it is discarded because it will always
        # be classified as non-indicative
        if edge not in index:
            return None
        path_vector[index[edge]] = count
    return path_vector


def load_whitelist(file_path, schema):
    """Loads a predefined whitelist
    :param file_path The whitelist's file
    :param schema The edge types mapping from name to ID
    """
    whitelist = np.zeros(len(schema))

    with open(file_path) as in_file:
        for line in in_file:
            edge_type = line.strip()
            if edge_type in schema:
                whitelist[schema.index(edge_type)] = 1

    # Add the empty path, to classify identical pairs as positive
    if '$' in schema:
        whitelist[schema.index('$')] = 1
    else:
        print 'warning: identity edge type is not in the whitelist'

    return whitelist
