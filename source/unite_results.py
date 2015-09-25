import re
import sys
import optparse

pairs = {}

def main():

    """
    Merges path files of the same dataset for different resources.
    """

    # Get the arguments - the path files of every resource,
    # and last, the output file path
    opt_parser = optparse.OptionParser()
    (opts, args) = opt_parser.parse_args()

    # Load all path files
    map(load_pairs, args[:-1])

    # Print the pairs with their annotation
    with open(args[-1], 'w') as output:
        for pair_num, key in enumerate(pairs.keys()):
            output.write('pair number ' + str(pair_num + 1) + ': ' + key + '\n')
            if len(pairs[key]) > 0:
                output.write('\n'.join(pairs[key]) + '\n')


# Load the paths of the pairs
def load_pairs(paths_file):

    global pairs
    pair_pattern = re.compile('pair number [0-9]+: (.+)->(.+)')

    with open(paths_file) as f:
        for line in f:
            match = pair_pattern.search(line.strip())

            # Pair line
            if match is not None:
                curr_xy = '->'.join([match.group(1), match.group(2)])
                if curr_xy not in pairs:
                    pairs[curr_xy] = []

            # Path line
            else:
                pairs[curr_xy].append(line.strip())

    return pairs


if __name__ == '__main__':
    main()