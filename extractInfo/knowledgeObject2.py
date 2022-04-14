
class KnowledgeObject:

    def __init__(self):
        self.ID = None
        self.TITLE = None
        self.DOC = None
        self.PAIR = ()
        self.MENTION = None
        self.REL = None

    def save(self):
        pass

    def get_pos(self, e):
        for index, token in enumerate(self.DOC):
            if token == entity:
                pos = self.DOC[index].pos_
                return pos

    def get_label(self, e):
        for index, entity in enumerate(self.DOC.ents):
            if e == entity:
                label = self.DOC.ents.label_
                return label

    def as_row(self):
        e1 = self.PAIR[0]
        e2 = self.PAIR[-1]
        pos1 = self.get_pos(e1)
        pos2 = self.get_pos(e2)
        label1 = self.get_label(e1)
        label2 = self.get_label(e2)
        rel = self.REL
        return [rel, e1, e2, pos1, pos2, label1, label2]
