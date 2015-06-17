import numpy as np

from aprf import multi_calc_aprf


class HardPathValidator:
    """
    Binary edge model -
    classifies a path as indicative if it contains only edge types in the whitelist,
    and classifies a term-pair as positive if at least one of its paths is indicative.
    """

    def __init__(self, f_beta_sq=None, regularization=1.0, num_epochs=100, population_ratio=1.0, p_mutation=0.1):
        """
        Initialize the binary edge model.
        :param f_beta_sq: the beta parameter for F_beta^2 measure
        :param regularization: the regularization factor (lambda)
        :param num_epochs: the number of generations of the genetic search
        :param population_ratio: population size / number of edge types
        :param p_mutation: the mutation probability
        """
        self.f_beta_sq = f_beta_sq
        self.regularization = regularization
        self.num_epochs = num_epochs
        self.population_ratio = population_ratio
        self.p_mutation = p_mutation

    def learn(self, path_sets, labels):
        """Train the binary edge model
        :param path_sets -- the term-pairs represented as path-sets
        :param labels -- the term-pairs gold standard annotations
        """

        pair_to_paths, path_features = self.prebuild_data_structures(path_sets)
        num_features = path_features.shape[1]

        f_beta_q = self.f_beta_sq
        regularization = self.regularization
        num_epochs = self.num_epochs
        population = int(num_features * self.population_ratio)
        p_mutation_init = self.p_mutation
        p_mutation = p_mutation_init

        # Create the initial population randomly
        white_lists = np.random.choice(2, size=(population, num_features))

        # Evaluate the fitness function for every whitelist in the population
        f = lambda wl: fitness(pair_to_paths, path_features, wl, labels, f_beta_q, regularization)

        # Create a new generation
        for epoch in xrange(num_epochs):
            scores = f(white_lists)

            # Choose a pair of whitelists for crossover
            males = white_lists[np.random.choice(population, size=population, p=scores)]
            females = white_lists[np.random.choice(population, size=population, p=scores)]

            mating_points = np.random.choice(num_features - 1, size=population) + 1

            for i, mp in enumerate(mating_points):
                males[i, mp:] = 0
                females[i, :mp] = 0
            white_lists = males + females

            # Create a mutation with probability p_mutation
            mutations = np.random.choice(2, size=(population, num_features), p=[1.0 - p_mutation, p_mutation])
            white_lists = np.abs(white_lists - mutations)

            # Reduce the mutation probability
            p_mutation = p_mutation_init * (1.0 - (float(epoch) / num_epochs))

        # Choose the best whitelist in the last generation
        self.white_list = white_lists[np.argmax(f(white_lists))]

    def classify(self, path_sets):
        """Classify each of the pairs in path_sets using the whitelist"""
        pair_to_paths, path_features = self.prebuild_data_structures(path_sets)
        return classify(pair_to_paths, path_features, self.white_list)

    def prebuild_data_structures(self, path_sets):
        """Create a matrix of paths and an index from each term-pair
        to its paths.
        :param path_sets: the original data structure of the pairs representation
        """
        i = 0
        pair_to_paths = []
        for path_set in path_sets:
            pair_to_paths.append(range(i, i + len(path_set)))
            i += len(path_set)

        path_features = [path for path_set in path_sets for path in path_set]
        path_features = np.vstack(path_features)

        return pair_to_paths, np.array(path_features, dtype=np.int)


class HardPathValidatorWhitelist(HardPathValidator):

    def __init__(self, whitelist):
        """
        Initialize the binary edge model with a predefined whitelist.
        :param whitelist: The list of allowed edge types
        """
        self.white_list = whitelist


def classify(pair_to_paths, path_features, white_lists):
    """
    Classify each of the pairs in path_sets using the whitelist
    :param pair_to_paths: the paths' indices for each term-pair
    :param path_features: the paths matrix
    :param white_lists: the whitelists used for classification
    """
    if len(white_lists.shape) == 1:
        population = 1
    else:
        population = white_lists.shape[0]

    # Classify a path as indicative if all its edge types are in the whitelist
    path_classifications = 1 - np.sign(path_features.dot(1 - white_lists.T))

    if population == 1:
        predictions = [0 if len(path_set) == 0 else
                       np.sign(np.sum(path_classifications[path_set])) for path_set in pair_to_paths]
    else:
        predictions = [np.zeros(population) if len(path_set) == 0 else
                       np.sign(np.sum(path_classifications[path_set], axis=0)) for path_set in pair_to_paths]
    return np.array(predictions)


def fitness(pair_to_paths, path_features, white_lists, labels, f_beta_q, regularization):
    """
    Compute the fitness of each whitelist: the F_beta score on the training set,
    subject to L2 regularization.
    :param pair_to_paths: the paths' indices for each term-pair
    :param path_features: the paths matrix
    :param white_lists: the whitelists used for classification
    :param labels: the term-pairs gold standard annotations
    :param f_beta_q: the beta parameter for F_beta measure
    :param regularization: the lambda regularization factor
    :return:
    """
    num_features = white_lists.shape[1]
    predictions = classify(pair_to_paths, path_features, white_lists)
    if f_beta_q is None:
        results = multi_calc_aprf(labels, predictions)[0]
    else:
        results = multi_calc_aprf(labels, predictions, f_beta_q)[3]
    scores = results - regularization * np.sum(white_lists, axis=1) / num_features
    scores[scores < 0] = 0

    # The probability to choose every whitelist is score^2/sum(score)
    scores = scores * scores
    norm = np.sum(scores)
    if norm > 0:
        return scores / norm
    else:
        return np.ones(len(scores)) / len(scores)
