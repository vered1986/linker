import re
import sys
import optparse
    
# Load the paths of the pairs
def load_pairs(paths_file):
    
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

# Get the arguments
optparser = optparse.OptionParser()
(opts, args) = optparser.parse_args()

# Load all results files
pairs = {}

for results_file_name in args[:-1]:
    load_pairs(results_file_name)

# Print the pairs with their annotation
pair_num = 1
with open(args[-1], 'w') as output:
    for key in pairs.keys():
        output.write('pair number ' + str(pair_num) + ': ' + key + '\n')
        pair_num = pair_num + 1
        for path in pairs[key]:
            output.write(path + '\n')