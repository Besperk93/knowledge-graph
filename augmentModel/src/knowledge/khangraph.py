import pandas as pd
import torch
import torch.nn as nn
import os
import re
from src.model.knowledgeBase import KnowledgeBase, KnowledgeBaseRegistry
from .embeddedKnowledge import EmbeddedKnowledge
from .tripleStore import TripleStore


@KnowledgeBaseRegistry.instance.register('khangraph')
class KhanGraph(KnowledgeBase):

    def __init__(self, embedding_path="/home/besperk/Code/knowledge-graph/augmentModel/src/knowledge/output/", data_path="/home/besperk/Code/knowledge-graph/Vault/graph/"):
        super(KhanGraph, self).__init__(embedding_dimension=200)
        # Create embedded knowledge
        self.embedder = EmbeddedKnowledge(self.embedding_dimensions, self.pad_id)
        # TODO: This will need to be reworked into something that loads the csv's and creates a dataframe object
        self.embedder.load_embedding(embedding_path)
        self.graph = TripleStore(data_path)

    def save(self, save_directory:str) -> None:
        """Save knowledge base in directory"""
        self.embedder.save(save_directory)

    @property
    def config(self) -> dict:
        #NOTE: Not sure if this is neccessary
        # get base configuration and add to it
        config = super(KhanGraph, self).config
        # return
        return config

    @staticmethod
    def load(directory:str, config:dict):
        """Load from saved directory"""
        return KhanGraph(directory)

    def find_entity_mentions(self, tokens):
        """Return a dictionary that maps a mention term to it's token IDs"""
        # string = ''.join(tokens)
        existing = [self.graph.check_entity(token) for token in tokens]
        return {tokens[i]: e for i, e in enumerate(existing) if e is not None}


    def find_candidates(self, entity):
        """Get candidate entities from a mention, return a list of entity IDs that get passed to the embed function"""
        candidates = self.graph.get_entity_candidates(entity, 10)
        return candidates

    def find_embedding(self, entity1s):
        """Embed the given entities based on their IDs, return a tensor"""
        existing = [self.graph.check_entity(token) for e1 in entity1s]
        embedding_ids = [self.embedder.embedding_map.get(w, 0) if w is not None else self.pad_id for w in existing]
        embedding_tensor = torch.tensor(embedding_ids).long().view(entity1s.size())
        return self.embedder.embedded(embedding_tensor)



    def get_prior(self, candidate_id):
        """Get probability for a given entity"""
        pass
