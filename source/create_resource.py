import sys
from collections import Counter
from matrix_serializer import save_matrix, read_mm_matrix

triplets_file = sys.argv[1]
out_prefix = sys.argv[2]

if len(sys.argv) > 3:
    prop_min_occ = int(sys.argv[3])
else:
    prop_min_occ = 0

with open(triplets_file) as input:
        dataset = set((tuple(line.strip().split('\t')) for line in input))
        prop_count = Counter((statement[2] for statement in dataset))

# Get top prop_cutoff properties
properties = sorted(prop_count.items(), key=lambda t: t[1], reverse=True)

properties = [prop for prop in properties if prop[1] >= prop_min_occ]

properties = set(zip(*properties)[0])
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