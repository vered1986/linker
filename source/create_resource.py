import sys
from collections import Counter
from matrix_serializer import save_matrix, read_mm_matrix
from common_data import load_data, load_whitelist
from docopt import docopt

args = docopt("""Creates the resource files from a triplets file.

    Usage:
        create_resource.py <triplets_file> <out_prefix>

        <out_prefix> = the directory and prefix for the resource files
        (e.g. /home/user/resources/res1).
    """)

triplets_file = args['<triplets_file>']
out_prefix = args['<out_prefix>']

with open(triplets_file) as input:
        dataset = set((tuple(line.strip().split('\t')) for line in input))
        properties = set((statement[2] for statement in dataset))

prop_id_to_label = sorted(properties)
prop_label_to_id = dict(((p, i) for i, p in enumerate(prop_id_to_label)))

dataset = [statement for statement in dataset if (len(statement[0]) > 0) and (len(statement[1]) > 0) and (statement[2] in properties)]

lhs_all, rhs_all, prop_all = zip(*dataset)
entities = set(lhs_all).union(set(rhs_all))
ent_id_to_label = sorted(entities)
ent_label_to_id = dict(((e, i) for i, e in enumerate(ent_id_to_label)))

with open(out_prefix + '-l2r.txt', 'w') as output:
        for lhs, rhs, prop in dataset:
                lhs_id = ent_label_to_id[lhs]
                rhs_id = ent_label_to_id[rhs]
                prop_id = prop_label_to_id[prop]
                print >>output, '\t'.join(map(str, (lhs_id, rhs_id, prop_id)))

# Save the properties map
with open(out_prefix + 'Properties.txt', 'w') as output:
        for i, p in enumerate(prop_id_to_label):
                print >>output, '\t'.join(map(str, (i, p)))

# Save the entities map
with open(out_prefix + 'Entities.txt', 'w') as output:
        for i, e in enumerate(ent_id_to_label):
                print >>output, '\t'.join(map(str, (i, e)))

# Create the matrix
mm_file = out_prefix + 'Matrix.mm'
with open(mm_file, 'w') as output:
    print >>output, '%%MatrixMarket matrix coordinate integer general'
    print >>output, ' '.join(map(str, (len(entities), len(entities), 0, len(properties))))

    with open(out_prefix + '-l2r.txt') as input:
        for line in input:
            lhs, rhs, prop = line.strip().split('\t')
            print >>output, ' '.join(map(str, (int(lhs) + 1, int(rhs) + 1, 1)))


mm_matrix = read_mm_matrix(mm_file)
save_matrix(mm_file + '.tmp', mm_matrix)