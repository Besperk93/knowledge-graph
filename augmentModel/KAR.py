"""
 Entity linking:
   Input = sequence of tokens
   Output = list of spans + entity id of linked entity
       span_indices = (batch_size, max_num_spans, 2)
       entity_id = (batch_size, max_num_spans)

 Proceeds in two steps:
   (1) candidate mention generation = generate a list of spans and possible
       candidate entitys to link to
   (2) disambiguated entity to predict


Model component is split into several sub-components.

 Candidate mention generation is off loaded to data generators, and uses
 pre-processing, dictionaries, and rules.

 EntityDisambiguation: a module that takes contextualized vectors, candidate
   spans (=mention boundaries, and candidate entity ids to link to),
   candidate entity priors and returns predicted probability for each candiate.

 EntityLinkingWithCandidateMentions: a Model that encapusulates:
   a LM that contextualizes token ids
   a EntityDisambiguation that predicts candidate mentions from LM context vectors
   a loss calculation for optimizing
   (optional) a KG embedding model that can be used for multitasking entity
        embeddings and the entity linker
"""

from transformers import AutoTokenizer


class KARComponent:



    def predict_candidates(self):
        pass

    def query_kb(self):
        pass

    def load_kb(self):
        pass

    def handle_input(self):
        pass

    def __init__(self, kb):
        self.KB = kb
        self.TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
