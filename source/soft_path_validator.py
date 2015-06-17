import numpy as np


class SoftPathValidator:
    """
    Weighted edge model -
    classifies a path according to a score based on the weight of its edge types,
    and classifies a term-pair as positive if at least one of its paths is indicative.
    """

    def __init__(self, f_beta_sq=None, regularization=1.0, learning_rate=1.0, num_epochs=100):
        """
        Initialize the weighted edge model.
        :param f_beta_sq: the beta parameter for F_beta measure
        :param regularization: the regularization factor (lambda)
        :param learning_rate: learning rate (for w's update)
        :param num_epochs:  the number of epochs
        """
        self.f_beta_sq = f_beta_sq
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def learn(self, path_sets, labels):
        """Train the weighted edge model
        :param path_sets -- the term-pairs represented as path-sets
        :param labels -- the term-pairs gold standard annotations
        """
        regularization = self.regularization
        num_epochs = self.num_epochs

        # Compute alpha = 1/(1+beta^2) - for F measure
        alpha = None if self.f_beta_sq is None else np.reciprocal(self.f_beta_sq + 1.0)

        lr_init = self.learning_rate
        lr = lr_init

        num_pairs = len(labels)

        pair_to_paths, path_features = prebuild_data_structures(path_sets)

        # Initialize the edge types weights randomly
        w = np.random.normal(size=path_features.shape[1])

        # Choose a maximum-score path for each term-pair randomly
        max_paths = np.zeros(num_pairs, dtype=np.int)
        for i, (path_set, label) in enumerate(zip(pair_to_paths, labels)):
            max_paths[i] = np.random.choice(path_set)

        for epoch in xrange(num_epochs):

            # "M-step": Use F_beta derivative to update w
            f = path_features[max_paths]

            predictions = probabilities(w, f)

            dp = (f.T * (predictions * (1.0 - predictions))).T    # Matrix of the same shape as "features"
            sum_dp = np.sum(dp, axis=0)                           # Vector (len = num features)
            sum_g_dp = labels.dot(dp)                             # Vector (len = num features)
            sum_p = np.sum(predictions)                           # Scalar
            sum_g = np.sum(labels)                                # Scalar
            sum_g_p = labels.dot(predictions)                     # Scalar

            denominator = np.reciprocal(alpha*sum_p + (1 - alpha)*sum_g)

            dw = denominator * sum_g_dp - alpha * denominator * denominator * sum_g_p * sum_dp
            dw *= num_pairs

            w += lr * (dw - regularization*w)

            # Reduce the learning rate
            lr = lr_init * (1.0 - (float(epoch) / num_epochs))

            # "E-step": Give a score to each path according to the current weights,
            # and choose the highest-scored path for each pair
            path_probabilities = probabilities(w, path_features)
            for i, (path_set, label) in enumerate(zip(pair_to_paths, labels)):
                max_paths[i] = path_set[np.argmax(path_probabilities[path_set])]

        self.weights = w

    def classify(self, path_sets):
        """Classify each of the pairs in path_sets using the weights
        vector and the path with the highest score"""
        pair_to_paths, path_features = prebuild_data_structures(path_sets)
        path_probabilities = probabilities(self.weights, path_features)

        predictions = []
        for path_set in pair_to_paths:
            path_p = path_probabilities[path_set]
            pair_p = np.max(path_p)
            pair_c = np.sign(np.sign(pair_p - 0.5) + 1)
            predictions.append(pair_c)
        return np.array(predictions)


def prebuild_data_structures(path_sets):
    """Create a matrix of paths and an index from each term-pair to its paths.
    :param path_sets: the original data structure of the pairs representation
    """
    i = 0
    pair_to_paths = []
    for path_set in path_sets:
        pair_to_paths.append(range(i, i + len(path_set)) + [-1])
        i += len(path_set)

    path_features = [path for path_set in path_sets for path in path_set]
    path_features = np.vstack(path_features)

    # Add the default empty path
    path_features = np.vstack([path_features, np.zeros(path_features.shape[1])])

    # Add a bias to all paths' feature vectors
    return pair_to_paths, add_bias(path_features)


def probabilities(weights, features):
    """Return the paths' scores = sigmoid(w * path features)"""
    scores = weights.dot(features.T)
    return np.array(map(sigmoid, scores))


def add_bias(features):
    """Add a bias to each path features"""
    return np.hstack([features, np.ones((features.shape[0], 1))])


def sigmoid(x):
    """Sigmoid(x) = 1 / (1 + e^-x)"""
    if x <= -MAX_VAL:
        return 0.0
    elif x >= MAX_VAL:
        return 1.0
    else:
        return sigmoid_table[int((x + MAX_VAL)*STEP)]


MAX_VAL = 6.0
EXP_TABLE_SIZE = 1000
STEP = EXP_TABLE_SIZE / (2*MAX_VAL)
sigmoid_table = np.reciprocal(np.exp(-np.linspace(-MAX_VAL, MAX_VAL, EXP_TABLE_SIZE)) + 1)