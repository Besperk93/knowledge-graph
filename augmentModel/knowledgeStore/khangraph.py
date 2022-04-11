from .knowledgeBase import knowledgeBase
from embeddedKnowledge import embeddedKnowledge


@KnowledgeBaseRegistry.instance.register('khangraph')
class KhanGraph(KnowledgeBase):

    def __init__(self, data_path="../Vault/working-graph"):
        super(KhanGraph, self).__init__(embedding_dimension=200)
        # Create embedded knowledge
        self.embedder = embeddedKnowledge(self.embedding_dimensions, self.pad_id)
        # NOTE: This will need to be reworked
        self.embedder.load(data_path)

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
