import torch
import transformers
from model import KnowGPT2Model, KnowBertForPretraining
from knowledgeBase import KnowledgeBase, KnowledgeBaseRegistry

@KnowledgeBaseRegistry.instance.register("example-test-kb")
class TestKB(KnowledgeBase):
    """ Simple af knowledge base """

    def __init__(self):
        super(TestKB, self).__init__(embedd_dim=12)

    def find_mentions(self, tokens):
        # hard coded mentions for both examples
        if tokens[0].lower() == "this":
            return {"nice_coffee": [3, 4]}
        if tokens[0].lower() == "i":
            return {"like": [3], "place": [5], "very_much": [6, 7]}

    def find_candidates(self, mention):
        # hard code candidate ids
        # different number of candidates for different mentions
        if mention == "like":
            return [1, 1]
        return [1]

    def get_prior(self, entity_id):
        return 1

    def embedd(self, entity_ids):
        # random entity embedding
        return torch.empty((*entity_ids.size(), self.embedd_dim)).uniform_(-1, 1)

# sample sentences
sampleA = "This is a nice coffee spot and the food was tasty too!"
# sampleB = "I do not like this place very much!"

# create bert model
gpt2 = KnowGPT2Model.from_pretrained("gpt2")
# add knowledge base
kb_A = gpt2.add_kb(3, TestKB())
# kb_B = bert.add_kb(2, TestKB())

# create tokenizer
tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

tokensA = tokenizer.tokenize(sampleA)
# tokensB = tokenizer.tokenize(sampleB)
# print(f"A: {tokensA}\nB: {tokensB}")

input_ids_A = tokenizer(sampleA)['input_ids']
# input_ids_B = tokenizer(sampleB)['input_ids']
# print(f"A: {input_ids_A}\nB: {input_ids_B}")

try:
    # create tensor
    input_ids = torch.tensor([input_ids_A]).long()
except Exception as e:
    print(f"Could not create input_ids: {repr(e)}")


# prepare and execute
gpt2.prepare_kbs([tokensA])
output = gpt2.forward(input_ids)
print(output)
