import pickle
import pandas as pd
import re

loc = "./Vault/mtb/output/relations.pkl"

with open(loc, 'rb') as file:
    train = pickle.load(file)

print(train.idx2rel)


# string = r'9 "The rain in spain falls mainly on the plain"'
# print(re.match("^\d+", string)[0])
