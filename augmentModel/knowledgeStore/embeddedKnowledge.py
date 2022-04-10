import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from collections import OrderedDict



class EmbeddedKnowledge:
    # Contains a map of words to embeddings

    def __init__(self, embedding_dimensions, pad_id=None):
        self.pad_id = pad_id
        self.embedding_dimensions = embedding_dimensions
        self.embedding_map = None
        self.embedding:nn.Embedding = None

    def embed(self, ids):
        # Check an embedding has been loaded before use
        assert self.embedding is not None
        ids[ids == self.pad_id] = 0
        return self.embedding(ids)

    def load_embedding(self, word_path, weight_path):
        # Load a trained embedding from a file
        with open(path, 'r') as embedding_words:
            words = embedding_words.read().split('\n')
            self.word2id = {word: i for i, word in enumerate(words, 1)}
        weight = torch.load(weight_path)
        assert weight.size(1) == self.embedding_dimensions
        weight = torch.cat((torch.zeros((1, self.embedding_dimensions)), weight), dim=0)
        self.embedding = nn.Embedding(num_embeddings=weight.size(0), embedding_dim=self.embedding_dimensions, _weight=weight)

    def train_embedding(self, graph_loc, model="TransE"):
        # train an embedding from a provided knowledge graph
        # Get triples from csv
        df = pd.read_csv(graph_loc)
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
        # Create triple factory
        train_factory = TriplesFactory(triples=train_triples)
        test_factory = TriplesFactory(triples=test_triples)
        # Run pipeline
        results = pipeline(
            training_triples_factory=train_factory,
            testing_triples_factory=test_factory,
            model=model,
            model_kwargs={
                "embedding_dim": self.embedding_dimensions,
                "automatic_memory_optimization": True
            })
        weight = results.model.entity_embeddings.weight.cpu()
        # Update word2id map
        self.word2id = OrderedDict(zip(e1s, range(1, len(e1s) + 1)))
        self.embedding = nn.Embedding(
            num_embeddings=len(e1s) + 1,
            embedding_dim=self.embedding_dimensions,
            _weight = torch.cat((torch.zeros((1, self.embedding_dimensions)), weight), dim=0)
            )
        return results


    def save_embedding(self, output_loc):
        # Save embedding for reuse
        torch.save(self.embedding.weight[1:, ...], dump_path)
        with open(dump_path, 'w+') as output:
            f.write('\n'.join(self.word2id.keys()))
