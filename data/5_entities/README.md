1. this is a results folder: holds entities and relations.

Final output from the pipeline - contains:
- Individual .entities.txt files
- Consolidated knowledge graph files (__ALL_ENTITIES_ALL_RELATIONS_PARALLEL.txt)

To consolidate existing *.entities.txt files in this folder:
python 4b_batch_parallel_extract_entities_from_augmented_txts.py data/5_entities/

To consolidate from custom entities folder:
python 4b_batch_parallel_extract_entities_from_augmented_txts.py /path/to/entities/folder/
