
class KnowledgeObject:

    """Graph compatable knowldege object"""

    def __init__(self, name):
        self.ID = None
        self.NAME = name
        self.POS = None
        self.LABEL = []
        self.RELATIONS = {}
        self.MENTION = None

    def save_object(self):
        pass

    def get_links(self, e_filter=None, r_filter=None):
        if e_filter:
            neighbours = [r for r, e in self.RELATIONS.items() if e.LABEL == e_filter]
        elif r_filter:
            neighbours = [r for r, e in self.RELATIONS.items() if r.LABEL == r_filter]
        else:
            [r for r, e in self.RELATIONS.items()]
        return neighbours

    def get_neighbours(self, e_filter=None, r_filter=None):
        if e_filter:
            neighbours = [e for r, e in self.RELATIONS.items() if e.LABEL == e_filter]
        elif r_filter:
            neighbours = [e for r, e in self.RELATIONS.items() if r.LABEL == r_filter]
        else:
            [e for r, e in self.RELATIONS.items()]
        return neighbours

    def embedding(self, dimensions):
        pass
