# linker: Lexical INference using KnowledgE Resources

linker is an open source supervised framework for utilizing structured knowledge resources to recognize lexical inference.
If you use linker for any published research, please include the following citation:

"Learning to Exploit Structured Resources for Lexical Inference"
Vered Shwartz, Omer Levy, Ido Dagan and Jacob Goldberger. CoNLL 2015.

## Requirements ##
- Python 2.7 (with numpy and scipy)

## Usage Instructions ##
- Clone the repository or download the scripts.

- Search for paths between (x,y) terms in the datasets:
```
python search.py [dataset_path] [resource_matrix_path] [resource_entities_path] [resource_properties_path] [resource_l2r_path] [max_path_length] [allow_reversed_edges] [find_relevant_nodes (or use an existing nodes file)] [relevant_nodes_file] [paths_out_file]
``` For instance:
```
python search.py data/train_label /resource/wordnetMatrix.mm.tmp.npz /resource/wordnetEntities.txt /resource/wordnetProperties.txt /resource/wordnet-l2r.txt 1 True True data/nodes.txt data/train_path
```

- Train the model (binary / weighted) with the following command:
```
python -u eval_hard.py (eval_soft.py) [dataset] [eval_dir]
``` The eval_dir is the directory for the current evaluation. This directory should include a subdirectory called [dataset] that contains the train, test and validation sets. After evaluation, this directory will contain the path files, and the whitelists (in the binary model). This command evaluates the algorithm on this dataset using several Beta values and prints the performance results.

## Resources ##
Due to space limitations, only the WordNet resource is available in the repository. In order to use the other resources, download the resource dump files and create a triplets file in which every line is in the following format:
```
[left_object] [right_object] [property]
```
Use the script "create_resource.py" to create the resource files.
Using large resources requires sufficient memory.