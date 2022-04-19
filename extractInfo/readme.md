# Extracting Information #


This subdirectory contains the code used to extract information from the Khan Academy transcripts. It is structured as follows:

_annotatedReferences_

These contain annotated versions of scripts and functions from open sourced repositories. In this case from: https://github.com/plkmo/BERT-Relation-Extraction. Annotations of what each, relevant, function does were made as comments and proved important in understanding the processes involved. These are included to show where and how adaptations were made to suit this project's purposes.

_bin_

OLD STUFF THATS NOT USED, REMOVE THIS BEFORE UPLOADING TO MOODLE


_datasets_

Code used to generate the additions to the training corpus and the semeval dataset that we used to train and then fine tune the match the blanks BERT model. This code should output to the Vault.

_src_

Code used to define the models, training procedures, inference and processing involved in extracting information from the Khan Academy transcripts.
