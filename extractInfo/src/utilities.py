import os
import pickle
import re
import json
from itertools import permutations



def load_pickle(filename):
    """load data pickle from data directory"""
    with open(filename, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def save_as_pickle(filename, data):
    """save data as a pickle file"""
    with open(f".Vault/data/{filename}", 'wb') as output:
        pickle.dump(data, output)


def import_relations(loc1, loc2):
    with open(loc1, 'r') as in_file:
        rel2idx = json.load(in_file)
    with open(loc2, 'r') as in_file:
        idx2rel = json.load(in_file)
    relations = Relations_Mapper_From_Trained(rel2idx, idx2rel)
    return relations


# This is used in the preprocessing stages to prepare the blanked dataset "D".
def get_subject_objects(sent_):
    """return subject, object pairs parsed from dependency tree"""
    root = sent_.root
    subject = None
    objs = []
    pairs = []
    for child in root.children:
        if child.dep_ in ["nsubj", "nsubjpass"]:
            # The below will filter out all numbers and symbols - why?
            if len(re.findall("[a-z]+", child.text.lower())) > 0:
                subject = child
        elif child.dep_ in ["dobj", "attr", "prep", "ccomp"]:
            objs.append(child)
    if (subject is not None) and (len(objs) > 0):
        # Create subject object pairs for each possible permutation
        for a, b in permutations([subject] + [obj for obj in objs], 2):
            a_ = [w for w in a.subtree]
            b_ = [w for w in b.subtree]
            pairs.append((a_[0] if (len(a_) == 1) else a_, b_[0] if (len(_b) == 1) else b_))

    return pairs


def extract_mention(e1, e2, sentences, window):

    # Check if the entities are close enough together (window = 40) and if so extract the full span of the mention text
    if not (1 <= e2.start - e1.end <= window):
        return False

    # Find start of sentence
    punctuation = False
    start = e1.start - 1
    if start > 0:
        while not punctuation:
            punctuation = sentences[start].is_punct
            start -= 1
            if start < 0:
                break
        if start > 0:
            l_limit = start + 2
        else:
            l_limit = 0
    else:
        l_limit = 0


    # Find end of sentence
    punctuation = False
    start = e2.end
    if start < len(sentences) :
        while not punctuation:
            punctuation = sentences[start].is_punct
            start += 1
            if start == len(sentences):
                break
        if start < len(sentences):
            r_limit = start
        else:
            r_limit = len(sentences)
    else:
        r_limit = len(sentences)

    # Create list of tokens
    if (r_limit - l_limit) > window:
        return False
    else:
        x = [token.text for token in sentences[l_limit:r_limit]]
        r = (x, (e1.start - l_limit, e1.end - l_limit), (e2.start - l_limit, e2.end - l_limit))
        return r



class Relations_Mapper_From_Trained(object):
    def __init__(self, rel2idx, idx2rel):
        self.rel2idx = rel2idx
        self.idx2rel = idx2rel
