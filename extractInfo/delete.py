import pickle
import pandas as pd
import re
import json

loc = "./Vault/mtb/output/relations.pkl"

with open(loc, 'rb') as file:
    relations = pickle.load(file)

print(relations.idx2rel)

ids = relations.idx2rel
rels = relations.rel2idx

with open("./Vault/mtb/output/rel_map.json", 'w') as out:
    json.dump(rels, out)

with open("./Vault/mtb/output/id_map.json", 'w') as out:
    json.dump(ids, out)

# string = r'9 "The rain in spain falls mainly on the plain"'
# print(re.match("^\d+", string)[0])
