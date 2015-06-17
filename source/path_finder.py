class PathFinder:
    """
    Applies bi-directional search to find the shortest paths between every
    term-pair in the dataset.
    """

    # Consts
    L2R = 0
    R2L = 1
    FORWARD = 0
    BACKWARD = 1

    def __init__(self, knowledge_resource):
        """Init the relevant nodes search
        :param knowledge_resource -- the resource to use
        """

        self.knowledge_resource = knowledge_resource

    def find_shortest_paths(self, x, y, max_length, l2r_edges=None, r2l_edges=None):
        """Finds all the shortest paths between x and y subject to the maximum 
        length, using bi-directional search.
        :param x -- the ID of the first term
        :param y -- the ID of the second term
        :param max_length -- the maximum path length
        :param l2r_edges -- the left-to-right edges (when search space is reduced)
        :param r2l_edges -- the right-to-left edges (when search space is reduced)
        """

        # If the search space is not limited, the nodes can be expanded
        # according to all the edges in the resource graph
        if l2r_edges is None:
            l2r_edges = self.knowledge_resource.l2r_edges
        if r2l_edges is None:
            r2l_edges = self.knowledge_resource.r2l_edges
        allow_reversed_edges = self.knowledge_resource.allow_reversed_edges

        paths = []

        # If one of the entities is OOV, return an empty set
        if (x == -1) or (y == -1):
            return paths

        # Apply bi-directional search

        # Initialize the queues
        forward = {}  # Paths by the starting node
        forward[x] = [[]]  # Add the empty path
        backward = {}  # Paths by the ending node
        backward[y] = [[]]  # Add the empty path

        # Save the nodes that were already visited
        visited_forward = set()
        visited_backward = set()

        # Move one step in each direction until the root and goal meet
        for l in range(max_length + 1):

            intersection = set(forward.keys()).intersection(set(backward.keys()))

            # The root and goal met
            if len(intersection) > 0:
                return create_paths(forward, backward)

            # Make a step forward
            if l % 2 == 0:
                forward, visited_forward = expand(forward, l2r_edges, r2l_edges, visited_forward,
                                                  allow_reversed_edges, PathFinder.FORWARD)
            # Make a step backward        
            else:
                backward, visited_backward = expand(backward, r2l_edges, l2r_edges, visited_backward,
                                                    allow_reversed_edges, PathFinder.BACKWARD)

        return paths


def create_paths(forward, backward):
    """Connect the partial paths (edge types) in forward to those in backward
    where there is a mutual node
    :param forward -- a dictionary of nodes and paths starting in them
    :param backward -- a dictionary of nodes and paths ending in them
    """
    paths = []

    for node in set(forward.keys()).intersection(set(backward.keys())):
        for forward_path in forward[node]:
            for backward_path in backward[node]:
                path_type = forward_path + backward_path
                if path_type not in paths:
                    paths.append(path_type)

    return paths


def expand(dir_queue, dir_edges, opposite_dir_edges, visited_dir, allow_reversed_edges, expand_direction):
    """Expand the paths in a forward/backward step
    :param dir_queue -- a dictionary of nodes and paths starting/ending in them
    :param dir_edges -- the edges in the direction (left-to-right for forward, right-to-left for backward)
    :param opposite_dir_edges -- the edges in the opposite direction
    :param visited_dir -- the nodes that were already expanded in this direction
    :param allow_reversed_edges -- whether reversed edges are allowed in this resource
    """

    temp_dir_queue = {}

    # Get the next node from the queue and try to expand the path
    for node in dir_queue.keys():

        # Get the neighbors
        neighbors = []

        if node in dir_edges:
            neighbors = neighbors + [(neighbor, PathFinder.L2R, dir_edges[node][neighbor])
                                     for neighbor in dir_edges[node]]
        if allow_reversed_edges and (node in opposite_dir_edges):
            neighbors = neighbors + [(neighbor, PathFinder.R2L, opposite_dir_edges[node][neighbor])
                                     for neighbor in opposite_dir_edges[node]]

        # Don't visit a node twice in the same direction
        if node in visited_dir:
            continue

        visited_dir.add(node)

        # Get the right neighbors
        for (neighbor, direction, props) in neighbors:

            # If this neighbor is not visited
            if neighbor not in visited_dir:

                if neighbor not in temp_dir_queue.keys():
                    temp_dir_queue[neighbor] = []

                # Concatenate the edges to current paths
                for prop in props:
                    for path in dir_queue[node]:
                        if expand_direction == PathFinder.FORWARD:
                            new_path = path + [(prop, direction)]
                        else:
                            new_path = [(prop, direction)] + path
                        temp_dir_queue[neighbor].append(new_path)

    # Merge temp_forward into forward
    dir_queue = dict(temp_dir_queue.items() + dir_queue.items())
    return dir_queue, visited_dir
