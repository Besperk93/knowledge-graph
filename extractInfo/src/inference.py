import spacy
import torch
import os
import re
from transformers import BertModel, BertTokenizer
from utilities import load_pickle, import_relations
from itertools import permutations
from model.BERT.modeling_bert import BertModel as Model
from model.BERT.tokenization_bert import BertTokenizer as Tokenizer
from knowledgeObject import KnowledgeObject
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

class InferencePipeline:

    """Loads a trained model into an inference pipeline for extracting entities and relationships from an input transcript"""

    def __init__(self):
        if torch.cuda.is_available():
            self.CUDA = True
            self.DEVICE = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.NLP = spacy.load("en_core_web_trf")
        self.RM = import_relations("./Vault/mtb/output/rel_map.json", "./Vault/mtb/output/id_map.json")
        self.STOPWORDS = set(stopwords.words('english'))
        self.load_model()
        self.load_model_checkpoint()
        self.update_model()

    def extract_relations(self, doc, detect_entities=True):
        annotated_objs = self.annotate_sentences(doc)
        if annotated_objs != None:
            objs = []
            for obj in annotated_objs:
                # NOTE: sent contains an annotated entity sentence, can parse POS for each sent
                obj.REL = self.extract_one_relation(obj.MENTION)[-1]
                objs.append(obj)
            return objs

    def annotate_sentences(self, input):
        sentences = self.NLP(input)
        # Get list of objects from spacy nlp doc
        pairs = self.get_entities(sentences)
        pairs.extend(self.get_all_sub_obj_pairs(sentences))
        if len(pairs) == 0:
            # print("Did not find enough entities to extract relation")
            return
        annotated_list = []
        for obj in pairs:
            try:
                obj.DOC = sentences
                obj.MENTION = self.annotate_entity_mention(sentences, obj.PAIR[0], obj.PAIR[1])
                annotated_list.append(obj)
            except Exception as e:
                print(f"Error annotating entity mention: {repr(e)}")
        # Return a half-formed object
        return annotated_list

    def get_entities(self, input):
        try:
            entities = input.ents
            pairs = []
            if len(entities) > 1:
                for a, b in permutations(set([entity for entity in entities if not entity.text.lower() in self.STOPWORDS]), 2):
                    object = KnowledgeObject()
                    object.PAIR = (a, b)
                    pairs.append(object)
            return pairs
        except Exception as e:
            print(f"Error extracting entities: {repr(e)}")
            return

    def get_all_sub_obj_pairs(self, input):
        if isinstance(input, str):
            sents_doc = self.nlp(input)
        else:
            sents_doc = input
        # sent_ = next(sents_doc.sents)
        # root = sent_.root
        chunks = sents_doc.noun_chunks

        subject = None; objs = []; pairs = []
        for chunk in chunks:
            if chunk.root.dep_ in ["nsubj", "nsubjpass"]:
                #NOTE: Commenting this out to see if I can get better numerical info
                # if len(re.findall("[a-z]+",child.text.lower())) > 0: # filter out all numbers/symbols
                # NOTE: Replace with stopword filter
                if chunk.root.text.lower() in self.STOPWORDS:
                    continue
                else:
                    subject = chunk
            elif chunk.root.dep_ in ["dobj", "attr", "prep", "ccomp", "compound", "pobj", "quantmod"]:
                if chunk.root.text.lower() in self.STOPWORDS:
                    continue
                else:
                    objs.append(chunk)

        if (subject is not None) and (len(objs) > 0):
            for a, b in permutations(set([subject] + [obj for obj in objs]), 2):
                object = KnowledgeObject()
                object.PAIR = (a, b)
                pairs.append(object)

        return pairs

    def extract_one_relation(self, mention):
        self.NET.eval()
        tokenized = self.TOKENIZER.encode(mention, max_length=500)
        entity_starts = self.get_e1e2_start(tokenized)
        tokenized = torch.LongTensor(tokenized).unsqueeze(0)
        entity_starts = torch.LongTensor(entity_starts).unsqueeze(0)
        attention_mask = (tokenized != self.TOKENIZER.pad_token_id).float()
        token_type_ids = torch.zeros((tokenized.shape[0], tokenized.shape[1])).long()

        # Apply CUDA if available
        # Note: I'm not sure if the CUDA management is working optimally. Particularly for inference, a lot is still running on the CPU - could be something to do with relatively high dataloading and low tensor processing
        if self.CUDA:
            tokenized = tokenized.to(self.DEVICE)
            attention_mask = attention_mask.to(self.DEVICE)
            token_type_ids = token_type_ids.to(self.DEVICE)

        with torch.no_grad():
            classification_logits = self.NET(tokenized, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None, e1_e2_start=entity_starts)
            predicted = torch.softmax(classification_logits, dim=1).max(1)[1].item()

        relationship = self.RM.idx2rel[str(predicted)].strip()
        # print(f"Sentence: {mention}")
        # print(f"Relationship: {relationship} \n")
        return (mention, relationship)

    def get_e1e2_start(self, x):
        try:
            e1_e2_start = ([i for i, e in enumerate(x) if e == self.E1][0], [i for i, e in enumerate(x) if e == self.E2][0])
            return e1_e2_start
        except Exception as e:
            pass
            # print(f"Error getting entity start markers: {repr(e)}")
            # print(x)

    def annotate_entity_mention(self, sentences, e1, e2):
        annotated = ''
        e1start, e1end, e2start, e2end = 0, 0, 0, 0
        for token in sentences:
            # For the first entity
            if not isinstance(e1, list):
                if (token.text == e1.text) and (e1start == 0) and (e1end == 0):
                    annotated += ' [E1]' + token.text + '[/E1] '
                    e1start, e1end = 1, 1
                    continue
            else:
                if (token.text == e1[0].text) and (e1start == 0):
                    annotated += ' [E1]' + token.text + ' '
                    e1start += 1
                    continue
                elif (token.text == e1[-1].text) and (e1end == 0):
                    annotated += token.text + '[/E1] '
            # For the second entity
            if not isinstance(e2, list):
                if (token.text == e2.text) and (e2start == 0) and (e2end == 0):
                    annotated += ' [E2]' + token.text + '[/E2] '
                    e2start, e2end = 1, 1
                    continue
            else:
                if (token.text == e2[0].text) and (e2start == 0):
                    annotated += ' [E2]' + token.text + ' '
                    e2start += 1
                    continue
                elif (token.text == e2[-1].text) and (e2end == 0):
                    annotated += token.text + '[/E2] '
            annotated += ' ' + token.text + ' '
        annotated = annotated.strip()
        annotated = re.sub(' +', ' ', annotated)
        return annotated


    def update_model(self):
        try:
            # Note: Can we update the model continuously?
            self.NET.load_state_dict(self.CHECKPOINT['state_dict'])
            start_epoch = self.CHECKPOINT['epoch']
            best_pred = self.CHECKPOINT['best_acc']
            amp_checkpoint = self.CHECKPOINT['amp']
        except Exception as e:
            print(f"Error updating base model: {repr(e)}")
            return


    def load_model_checkpoint(self):
        try:
            self.CHECKPOINT = torch.load("./Vault/mtb/output/task_model_best.pth.tar")
        except Exception as e:
            print(f"Error loading trained checkpoint: {repr(e)}")
            return


    def load_model(self):
        try:
            self.NET = Model.from_pretrained("bert-base-uncased", model_size='bert-base-uncased', task='classification', n_classes_=26).to(self.DEVICE)
        except Exception as e:
            print(f"Error loading BERT from transformers: {repr(e)}")
            return
        try:
            self.TOKENIZER = Tokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
            self.TOKENIZER.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])
            self.E1 = self.TOKENIZER.convert_tokens_to_ids('[E1]')
            self.E2 = self.TOKENIZER.convert_tokens_to_ids('[E2]')
        except Exception as e:
            print(f"Error loading tokenizer: {repr(e)}")
            return
        try:
            self.NET.resize_token_embeddings(len(self.TOKENIZER))
        except Exception as e:
            print(f"Error resizing base model: {repr(e)}")
            return
