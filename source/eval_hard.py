from time import clock
import sys

import numpy as np

from aprf import calc_aprf
from hard_path_validator import HardPathValidator, HardPathValidatorWhitelist
from common_data import load_data, load_whitelist
from docopt import docopt


def main():
    """
    Reads the paths files and the datasets, trains the binary edge model
    and evaluates it.
    """

    args = docopt("""Trains and evaluates the binary model using several Beta values and prints the performance results.

    Usage:
        eval_hard.py <dataset> <eval_dir> [<whitelist>]

        <eval_dir> = directory for the current evaluation.
        <dataset> = a subdirectory with the train, test and validation sets.
        This is the output directory for path and whitelist files.
        <whitelist> = the whitelist file, in case of using a predefined whitelist.
    """)

    dataset_in = args['<dataset>']
    resource = args['<eval_dir>']
    whitelist = args['<whitelist>']

    time = clock()
    np.random.seed(17)

    for dataset in [dataset_in]:

        # Load a whitelist from a file
        if whitelist is not None:
            schema, test_path_sets, test_labels = load_data(resource + '/' + dataset + '/test')
            whitelist = load_whitelist(sys.argv[3], schema)
            validator = HardPathValidatorWhitelist(whitelist)

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

        identity_index = None
        if '$' in schema:
            identity_index = schema.index('$')

        for f_beta_sq in [0.00001] + [0.01*i for i in xrange(5, 200, 5)]:
            validators = [HardPathValidator(f_beta_sq=f_beta_sq, regularization=regularization, num_epochs=100,
                                            population_ratio=10.0, p_mutation=0.01, identity_index=identity_index)
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
                out.write('\n'.join([schema[edge_type]
                                     for edge_type in range(len(validator.white_list))
                                     if validator.white_list[edge_type] == 1]))

            print "%0.2f" % ((clock() - time) / 60),
            print dataset, 'Hard', f_beta_sq,
            print ' '.join(['%0.3f' % r for r in calc_aprf(test_labels, predictions)])
        print



if __name__ == '__main__':
    main()