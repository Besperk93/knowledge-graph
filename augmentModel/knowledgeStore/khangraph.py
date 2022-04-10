from .knowledgeBase import knowledgeBase
from embeddedKnowledge import embeddedKnowledge


class KhanGraph(KnowledgeBase):

    def __init__(self):
        pass

    @property
    def config(self) -> dict:
        config = super(KhanGraph, self).config
        return config

    def save(self):
        pass

    def load(self):
        pass

    def find_mentions(self, mention):
        # General method to find relevant mentions from input mention
        pass

    def find_token_mentions(self):
        # Specific method for this knowledge base
        # NOTE: will be working with BPE encoded tokens rather than wordpiece
        pass
