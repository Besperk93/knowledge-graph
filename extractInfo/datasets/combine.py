import os
import random

base_path = "./Vault/data/mtb_training/"

def combine_raw(loc1, loc2):
    """Combine the raw cnn pretraining data with the generated questions"""

    with open(os.path.join(base_path, loc1), 'r') as loc1:
        data1 = loc1.read()

    with open(os.path.join(base_path, loc2), 'r') as loc2:
        data2 = loc2.read()

    combined = data1 + data2

    with open(os.path.join(base_path, "combined_train.txt"), 'w') as out_file:
        out_file.write(combined)



combine_raw('cnn.txt', 'numeracy_pretrain.txt')

def combine_annotated(loc1, loc2):
    """combined the annoated questions with the semeval data"""

    with open(os.path.join(base_path, loc1), 'r') as loc1:
        data1 = loc1.read()

    with open(os.path.join(base_path, loc2), 'r') as loc2:
        data2 = loc2.read()    
