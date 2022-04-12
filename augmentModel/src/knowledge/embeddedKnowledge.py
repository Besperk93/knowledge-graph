import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from collections import OrderedDict


class EmbeddedKnowledge:
    # Contains a map of words to embeddings

    def __init__(self, embedding_dimension, pad_id=None):
        self.pad_id = pad_id
        self.embedding_dimension = embedding_dimension
        self.embedding_map = None
        self.embedding:nn.Embedding = None

    def embedded(self, ids):
        # Check an embedding has been loaded before use
        assert self.embedding is not None
        ids[ids == self.pad_id] = 0
        return self.embedding(ids)

    def load_embedding(self, path):
        print(f"Loading graph embedding from: {path}")
        # Load a trained embedding from a file
        with open(os.path.join(path, "embedding_words.txt"), 'r') as embedding_words:
            words = embedding_words.read().split('\n')
            self.embedding_map = {word: i for i, word in enumerate(words, 1)}
        with open(os.path.join(path, "embedding_weights.bin"), 'rb') as weights_bin:
            weight = pickle.load(weights_bin)
        assert weight.embedding_dim == self.embedding_dimension
        self.embedding = weight

    def train_embedding(self, triple_store, model="TransE"):
        # NOTE: Significant update required to get the pykeen modules to work, looks like the module is quite a few changes ahead of reference implementation
        # train an embedding from a provided knowledge graph
        # Get triples from csv
        df = triple_store.DF
        e1s = df["Entity1"].tolist()
        e2s = df["Entity2"].tolist()
        rs = df["Relation"].tolist()
        triples = list(zip(e1s, e2s, rs))
        triples = np.asarray(triples)
        n = len(triples)
        # prepare a test/train split
        train_mask = np.full(n, False)
        train_mask[:int(n * 0.9)] = True
        np.random.shuffle(train_mask)
        train_triples = triples[train_mask]
        test_triples = triples[~train_mask]
        # All Entities mapped to ids
        # all_entities = list(set([*e1s]))
        # all_relations = list(set(rs))
        # Create index map for each
        # entities_indexed = {i: e for i, e in enumerate(all_entities)}
        # relations_indexed = {i: e for i, e in enumerate(all_relations)}
        # Create triple factory
        train_factory = TriplesFactory.from_labeled_triples(
            triples=train_triples
            )
        test_factory = TriplesFactory.from_labeled_triples(
            triples=test_triples, entity_to_id=train_factory.entity_to_id, relation_to_id=train_factory.relation_to_id
            )
        # Run pipeline
        results = pipeline(
            training=train_factory,
            testing=test_factory,
            model=model,
            epochs=50,
            model_kwargs={
                "embedding_dim": self.embedding_dimension
            })
        weight = results.model.entity_representations[0]
        print(type(weight))
        # Update word2id map
        self.embedding_map = OrderedDict(zip(e1s, range(1, len(e1s) + 1)))
        self.embedding = weight._embeddings
        return results


    def save_embedding(self, output_loc):
        # Save embedding for reuse
        with open(os.path.join(output_loc, "embedding_weights.bin"), 'wb') as weights_bin:
            pickle.dump(self.embedding, weights_bin)
        with open(os.path.join(output_loc, "embedding_words.txt"), 'w+') as output:
            output.write('\n'.join(self.embedding_map.keys()))



if __name__ == '__main__':
    # import graph
    from tripleStore import TripleStore

    # load graph
    graph = TripleStore("/home/besperk/Code/knowledge-graph/Vault/working-graph/")
    # train a knowledge graph embedding for senticnet graph
    embedding = EmbeddedKnowledge(embedding_dimension=200)

    print("Training Embedding...")
    results = embedding.train_embedding(graph, model="TransE")
    eval_results = results.metric_results.to_flat_dict()
    for metric, value in eval_results.items():
        print(metric.ljust(30, ' '), value)

    print("Saving Embedding...")
    embedding.save_embedding("/home/besperk/Code/knowledge-graph/augmentModel/src/knowledge/bin/")
