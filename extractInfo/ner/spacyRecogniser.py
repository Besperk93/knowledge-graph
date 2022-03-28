import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")


class entityRecogniser:

    def run_ner(self):
        doc = nlp(self.TEXT)
        print(f'{len(doc.ents)} entities were found in this transcript')
        return [(token.text, token.label_) for token in doc.ents]

    def __init__(self, sentences):
        self.SENTENCES = sentences
        self.TEXT = ' '.join(sentences)
