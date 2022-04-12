from transformers import Tokenizer
from utilities import extract_mention
import spacy


class mtb_tokenizer:

    """Create triplets (statement, e1, e2), insert entity markers, blank entities when appropriate and tokenize for BERT"""

    def tokenize(self):

        self.create_triplets()
        return self.insert_markers()


    def insert_markers(self, D):

        self.TOKENIZER.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])

        # Unpack D
        (x, s1, s2), e1, e2 = D

        # Make sure all words are lower case
        x = [word.lower() for word in x if x != '[BLANK]']

        # Mask random words for MLM task
        forbidden_indexes = [i for i in range(s1[0], s1[1])] + [i for i in range(s2[0], s2[1])]
        permitted_indexes = [i for i in range(len(x)) if i not in forbidden_indexes]
        mask_ids = np.random.choice(permitted_indexes, size=round(self.MASKPB * len(permitted_indexes)), replace=False)
        x = [word if (idx not in mask_ids) else self.TOKENIZER.mask_token for idx, word in enumerate(x)]
        # Remember the masked words to pass as labels
        masked_words = [word.lower() for idx, word in enumerate(x) if (idx in mask_ids)]

        # Decide whether to blank an entity
        blank1 = np.random.uniform()
        blank2 = np.random.uniform()
        xBlank = ['[E1]', '[BLANK]', '[/E2]']
        if blank1 >= self.ALPHA:
            e1 = '[BLANK]'
        if blank2 >= self.ALPHA:
            e2 = '[BLANK]'

        # Mark entities and Blank words in input sequence
        if (e1 == '[BLANK]') and (e2 != '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]' ,'[BLANK]', '[/E1]'] + x[s1[1]:s2[0]] + ['[E2]'] + x[s2[0]:s2[1]] + ['[/E2]'] + x[s2[1]:] + [self.sep_token]

        elif (e1 == '[BLANK]') and (e2 == '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]' ,'[BLANK]', '[/E1]'] + x[s1[1]:s2[0]] + ['[E2]', '[BLANK]', '[/E2]'] + x[s2[1]:] + [self.sep_token]

        elif (e1 != '[BLANK]') and (e2 == '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]'] + x[s1[0]:s1[1]] + ['[/E1]'] + x[s1[1]:s2[0]] + ['[E2]', '[BLANK]', '[/E2]'] + x[s2[1]:] + [self.sep_token]

        elif (e1 != '[BLANK]') and (e2 != '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]'] + x[s1[0]:s1[1]] + ['[/E1]'] + x[s1[1]:s2[0]] + ['[E2]'] + x[s2[0]:s2[1]] + ['[/E2]'] + x[s2[1]:] + [self.sep_token]

        # Store the entity span starts to pass as labels
        e1_e2_start = ([i for i, e in enumerate(x) if e == '[E1]'][0], [i for i, e in enumerate(x) if e == '[E2]'][0])

        return x, masked_words, e1_e2_start


    def create_triplets(self):

        # Identify entities in raw text
        try:
            spacy.prefer_gpu()
            nlp = spacy.load("en_core_web_trf")
            sentences = nlp(self.RAW)
            entities = sentences.ents
        except Exception as e:
            print(f"Error extracting entities: {repr(e)}")

        # Create triplets from entities
        try:
            for i in range(len(entities)):
                e1 = entities[i]
                for j in range(1, len(entities) - i):
                    e2 = entities[i + j]
                    # Check the two entities are not the same
                    if e1.text.lower() == e2.text.lower():
                        continue
                    # Attempt to extract the mention
                    r = extract_mention(e1, e2, sentences, self.WINDOW)
                    if not r:
                        continue
                    else:
                        self.D.append((r, e1.text, e2.text))
        except Exception as e:
            print(f"Error creating triplets: {repr(e)}")


    def __init__(self, raw):
        self.TOKENIZER = Tokenizer.from_pretrained("bert-base-uncased")
        self.RAW = raw
        self.D = []
        self.WINDOW = 40
        self.ALPHA = 0.7
        self.MASKPB = 0.15
