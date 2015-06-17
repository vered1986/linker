from time import clock
import sys

import numpy as np

from aprf import calc_aprf
from hard_path_validator import HardPathValidator
from common_data import load_data, load_whitelist


def main():
    """
    Reads the paths files and the datasets, trains the binary edge model
    and evaluates it.

    :param sys.argv[1] -- the dataset file
    :param sys.argv[2] -- the directory of the path and label files
    """

    time = clock()
    np.random.seed(17)

    dataset_in = sys.argv[1]
    resource = sys.argv[2]

    for dataset in [dataset_in]:

        # Load a whitelist from a file
        if len(sys.argv) > 3:
            schema, test_path_sets, test_labels = load_data(resource + '/' + dataset + '/test')
            whitelist = load_whitelist(sys.argv[3], schema)
            validator = HardPathValidator(whitelist)

            # Evaluate on the test set
            predictions = validator.classify(test_path_sets)

            print dataset, 'Hard',
            print ' '.join(['%0.3f' % r for r in calc_aprf(test_labels, predictions)])
            return

        # Train the model using a range of betas and regularization lambdas
        schema, train_path_sets, train_labels = load_data(resource + '/' + dataset + '/train')
        schema, val_path_sets, val_labels = load_data(resource + '/' + dataset + '/val', schema)
        schema, test_path_sets, test_labels = load_data(resource + '/' + dataset + '/test', schema)

        # Change each path to a set of edges (load_data returns multiset)
        train_path_sets = map(np.sign, train_path_sets)
        val_path_sets = map(np.sign, val_path_sets)
        test_path_sets = map(np.sign, test_path_sets)

        for f_beta_sq in [0.00001] + [0.01*i for i in xrange(5, 200, 5)]:
            validators = [HardPathValidator(f_beta_sq=f_beta_sq, regularization=regularization, num_epochs=100,
                                            population_ratio=10.0, p_mutation=0.01)
                          for regularization in [0.01, 0.1, 1.0]]

            # Train the model
            [validator.learn(train_path_sets, train_labels) for validator in validators]

            # Choose the regularization by the validation set
            validator = validators[np.argmax([calc_aprf(val_labels, validator.classify(val_path_sets), f_beta_sq)[3]
                                              for validator in validators])]

            # Evaluate on the test set
            predictions = validator.classify(test_path_sets)

            # Save the whitelist to a file
            with open(resource + '/' + dataset + '/whitelist' + str(f_beta_sq), 'w') as out:
                for edge_type in range(len(validator.white_list)):
                    if validator.white_list[edge_type] == 1:
                        out.write(schema[edge_type] + '\n')

            print "%0.2f" % ((clock() - time) / 60),
            print dataset, 'Hard', f_beta_sq,
            print ' '.join(['%0.3f' % r for r in calc_aprf(test_labels, predictions)])
        print



if __name__ == '__main__':
    main()