# Knowledge Augmentation #

This subdirectory contains the code used to train a GPT2 model on the auxillary AMPS dataset, train an embedding for our knowledge base, integrated the knowledge base with the GPT2 model using a KAR component and then assess our models performance on the MATH dataset _and perhaps a GCSE maths paper_.

_src_

This directory contains the code used for the model, training procedures, KAR and evaluation of the KnowGPT2 model. The AMPS pretraining corpus is made up of _datasets_ created by the Khan_Academy.py and Mathematica_with_steps.py scripts, differening from the original paper in not including the _mathematica_ dataset. This divergence was due to memory constraints with the loading the full number of json files from the Vault.

Our knowldedge bases were constructed using the contents of the _knowledge_ directory, drawing from the raw csv's created by the _extractInfo_ directory and stored in the vault.

The _model_ directory contains the code used to create the KAR and the adapted GPT2 model.Scripts used to train and evaluate the model on the MATH Dataset are left in the parent directory (i.e. the train.py script).
