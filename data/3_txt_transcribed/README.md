1. this is a results folder: text files from transcription and document extraction.
2. to augment them with AI metadata, call:

# Augment with default directories (data/3_txt_transcribed/ -> data/4_augmented/):
python 3_batch_augment_transcripts.py

# Augment with custom directories:
python 3_batch_augment_transcripts.py data/3_txt_transcribed/ data/4_augmented/

# Using custom folder names:
python 3_batch_augment_transcripts.py /content/transcribed_docs/ /enriched/metadata_files/