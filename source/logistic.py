import numpy as np
from scipy.sparse import dok_matrix, hstack


class SoftLogistic:

    def __init__(self, f_beta_sq=None, regularization=1.0, learning_rate=1.0, num_epochs=100, batch_size=None):
        """Initialize the logistic regression.
        :param f_beta_sq: The beta value for F_beta^2 optimization (default None - optimize accuracy)
        :param regularization: The regularization Lambda (default 1.0)
        :param learning_rate: The learning rate (default 1.0)
        :param num_epochs: The number of epochs (default 100)
        :param batch_size: The batch size (default no batches)
        """
        self.f_beta_sq = f_beta_sq
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size


    def learn(self, features, labels):
        """Train the logistic regression model.
        :param features: The training instances' feature vectors
        :param labels: The training instances' labels
        """

        f = self.add_bias(features)
        w = np.random.normal(size=f.shape[1])

        regularization = self.regularization
        num_epochs = self.num_epochs

        # Optimize F_alpha for a suitable alpha (according to beta)
        alpha = None if self.f_beta_sq is None else np.reciprocal(self.f_beta_sq + 1.0)

        lr_init = self.learning_rate
        lr = lr_init

        batch_size = self.batch_size

        for epoch in xrange(num_epochs):
            if batch_size is None:
                f_batch = f
                labels_batch = labels
            else:
                batch = np.random.randint(0, f.shape[0], size=batch_size)
                f_batch = f[batch]
                labels_batch = labels[batch]

            predictions = np.array(map(sigmoid, f_batch.dot(w.T).T))

            if alpha is None:
                # Optimize Accuracy
                errors = labels_batch - predictions
                dw = f_batch.T.dot(errors.T).T

            else:
                # Optimize F_beta^2 
                dp = (f_batch.T.multiply(predictions * (1.0 - predictions))).T    # Matrix of the same shape as "features"
                sum_dp = dp.sum(axis=0)                                     # Vector (len = num features)
                sum_g_dp = dp.T.dot(labels_batch.T)                         # Vector (len = num features)
                sum_p = predictions.sum()                                   # Scalar
                sum_g = labels_batch.sum()                                  # Scalar
                sum_g_p = labels_batch.dot(predictions)                     # Scalar

                # Update rule for optimizing F_beta^2 - see paper for details
                denominator = np.reciprocal(alpha*sum_p + (1 - alpha)*sum_g)

                dw = denominator * sum_g_dp - alpha * denominator * denominator * sum_g_p * sum_dp
                dw *= len(labels_batch)
                if type(dw) != np.ndarray:
                    dw = np.array(dw)[0]

            w += lr * (dw - regularization*w)

            lr = lr_init * (1.0 - (float(epoch) / num_epochs))

        self.weights = w

    def classify(self, features):
        """Classify the instances
        :param features: feature vectors
        :return The binary classifications of each instance.
        """
        return np.sign(np.sign(self.probabilities(features) - 0.5) + 1)

    def probabilities(self, features):
        """Get the score of each instance.
        :param features: feature vectors
        :return The classification score of each instance.
        """
        features_with_bias = self.add_bias(features)
        scores = features_with_bias.dot(self.weights.T).T
        return np.array(map(sigmoid, scores))

    def add_bias(self, features):
        """Add the bias to each feature vector
        :param features: feature vectors
        """
        if type(features) == np.ndarray:
            return np.hstack([features, np.ones((features.shape[0], 1))])
        else:
            bias = dok_matrix((features.shape[0], 1))
            bias[:, 0] = 1
            bias = bias.tocsr()
            return hstack([features, bias], 'csr')


def sigmoid(x):
    # return np.reciprocal(np.exp(-x)+1)
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
