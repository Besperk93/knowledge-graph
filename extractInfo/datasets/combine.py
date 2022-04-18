import os
import random
import re

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

# combine_raw('cnn.txt', 'numeracy_pretrain.txt')


base_path = "./Vault/data/mtb_eval/"

def combine_annotated(loc1, loc2):
    """combined the annoated questions with the semeval data"""

    with open(os.path.join(base_path, loc1), 'r') as loc1:
        data1 = loc1.read()

    with open(os.path.join(base_path, loc2), 'r') as loc2:
        data2 = loc2.read()

    pattern = re.compile(r"^([0-9]+).*")
    lines = data1.split("\n\n") + data2.split("\n\n")
    random.shuffle(lines)
    print(lines[:3])
    re_indexed = [re.sub(r"^[0-9]+", str(i), line) for i, line in enumerate(lines)]
    print(re_indexed[:3])

    with open(os.path.join(base_path, "combined_eval.txt"), 'w') as out_file:
        for item in re_indexed:
            out_file.write(item + "\n\n")

combine_annotated("numeracy_test_annotated.txt", "semeval2010_task8/TRAIN_FILE.TXT")
