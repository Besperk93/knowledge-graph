
class KnowledgeObject:

    """Graph compatable knowldege object - WIP"""

    def __init__(self):
        self.ID = None
        self.NAME = None
        self.POS = None
        self.LABEL = []
        self.RELATIONS = {}
        self.MENTION = None

    def save_object(self):
        pass


    def add_entity(self, rel, e2):
        if self.RELATIONS.get(rel):
            self.RELATIONS.get(rel).append(e2)
        else:
            self.RELATIONS[rel] = [e2]

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
        # What if we could store the embedding within each object? Does that make sense?
        pass
