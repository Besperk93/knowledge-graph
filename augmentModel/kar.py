import torch
import torch.nn as nn

from transformers import BertConfig
from KAR.Utilities import set_start_weights, pseudo_inverse, match_shape_2d

import KAR.Recontextualizer as RR
import KAR.KnowledgeEnhancer as KE
import KAR.EntityLinker as EL
import KAR.MentionRepresenter as MR

class KAR(nn.Module):

    def __init__(self, kb, bert_config:BertConfig, span_encoder_config:dict, span_attention_config:dict, max_mentions:int, max_mention_span:int, max_candidates:int, threshold:float=None):

        super(KAR, self).__init__()
        span_encoder_config = BertConfig.from_dict(span_encoder_config)
        span_attention_config = BertConfig.from_dict(span_attention_config)

        # Establish mention parameters
        self.max_mentions = max_mentions
        self.max_mention_span = max_mention_span
        self.max_candidates = max_candidates

        # save knowledge base and create caches-list
        self.kb = kb
        self.cache = None
        self.clear_cache()

        # projections from bert to kb and reversed
        self.bert2kb = nn.Linear(bert_config.hidden_size, kb.embedding_dimensions)
        self.kb2bert = nn.Linear(kb.embedding_dimensions, bert_config.hidden_size)

        # create all modules
        self.mention_span_representer = MR.MentionRepresenter(self.kb, self.max_mentions)
        self.entity_linker = EL.EntityLinker(self.kb, span_encoder_config)
        self.enhanced_representation = KE.KnowledgeEnhancer(self.kb, threshold)
        self.recontextualizer = RR.Recontextualizer(self.kb, span_attention_config)

        # layernorms and dropout
        self.dropout = nn.Dropout(0.1)
        self.output_ln = nn.LayerNorm(bert_config.hidden_size)

        # Set initial Weights
        set_start_weights(self.output_ln, 0.02)
        set_start_weights(self.bert2kb, 0.02)
        with torch.no_grad():
            # get weight of first projection and compute it's pseudo-inverse
            w = self.bert2kb.weight.data
            w_inv = pseudo_inverse(w)
            # apply w-inv to bias
            b = self.bert2kb.bias.data
            b_inv = w_inv @ b
            # update parameters
            self.kb2bert.weight.data.copy_(w_inv)
            self.kb2bert.bias.data.copy_(b_inv)

    def get_cache_and_mention_candidates(self, tokens):
        # get mentions and mention spans
        mentions = self.kb.find_mentions(tokens)
        mention_terms, mention_spans = zip(*mentions.items()) if len(mentions) > 0 else ([], [])
        mention_terms, mention_spans = mention_terms[:self.max_mentions], mention_spans[:self.max_mentions]
        n_mentions = len(mention_terms)

        # max-size for dimension 1
        shape = (self.max_mentions, max(self.max_mention_span, self.max_candidates))
        # build mention-spans tensor
        mention_spans = match_shape_2d(mention_spans, shape, -1).long()
        # get candidate entity-ids for each found mention
        all_candidate_ids = [self.kb.find_candidates(m) for m in mention_terms]
        all_candidate_mask = [[1] * len(ids) for ids in all_candidate_ids]
        all_candidate_priors = [[self.kb.get_prior(i) for i in ids] for ids in all_candidate_ids]
        # match shape
        all_candidate_ids = match_shape_2d(all_candidate_ids, shape, self.kb.pad_id)
        all_candidate_mask = match_shape_2d(all_candidate_mask, shape, 0)
        all_candidate_priors = match_shape_2d(all_candidate_priors, shape, 0)

        # build mention-candidate map
        mention_candidate_map = list(zip(mention_terms, all_candidate_ids))
        # stack all tensors to build cache
        tensors = (mention_spans.float(), all_candidate_ids.float(), all_candidate_mask.float(), all_candidate_priors.float())
        cache = torch.stack(tensors, dim=0).unsqueeze(0)
        # return cache and mention-candidate-map
        return cache, mention_candidate_map

    def clear_cache(self):
        # empty cache
        self.cache = torch.empty((0, 4, self.max_mentions, max(self.max_mention_span, self.max_candidates))).float()

    def stack_caches(self, *caches):
        # stack all given caches on current cache
        self.cache = torch.cat((self.cache.to(caches[0].device), *caches), dim=0)

    def read_cache(self, cache=None):
        # read values from cache and convert to correct types
        mention_spans = (self.cache if cache is None else cache)[:, 0, :, :self.max_mention_span].long()
        candidate_ids = (self.cache if cache is None else cache)[:, 1, :, :self.max_candidates].long()
        candidate_mask = (self.cache if cache is None else cache)[:, 2, :, :self.max_candidates].bool()
        candidate_priors = (self.cache if cache is None else cache)[:, 3, :, :self.max_candidates].float()
        # clear cache after reading to prevent
        # false double use of the same cache
        self.clear_cache()
        # return values
        return mention_spans, candidate_ids, candidate_mask, candidate_priors

    def forward(self, h, cache=None):

        # read cache and move all to device
        mention_spans, candidate_ids, candidate_mask, candidate_priors = self.read_cache(cache)
        mention_spans, candidate_ids, candidate_mask, candidate_priors = mention_spans.to(h.device), candidate_ids.to(h.device), candidate_mask.to(h.device), candidate_priors.to(h.device)
        # check cache values
        if (len(mention_spans) != h.size(0)) or (candidate_ids.size(0) != h.size(0)) \
                or (candidate_mask.size(0) != h.size(0)) or (candidate_priors.size(0) != h.size(0)):
            raise RuntimeError("Cache size (%i) does not match batch-size (%i)!" % (len(mention_spans), h.size(0)))

        # get candidate-embeddings and build mention-mask
        candidate_embeddings = self.kb.embedd(candidate_ids).detach().to(h.device) # no gradients for entity embeddings
        mention_mask = ((mention_spans >= 0).sum(-1) > 0)

        # project hidden into entity-embedding space
        h_projections = self.bert2kb(h)

        # compute entity linking scores
        mention_span_representations = self.mention_span_representer(h_projections, mention_spans)
        linking_scores = self.entity_linker(candidate_embeddings, candidate_mask, candidate_priors, mention_span_representations, mention_mask)

        # Calculate Enhanced Representations
        enhanced_span_representations, entropy = self.enhanced_representation(linking_scores, candidate_embeddings, mention_span_representations, mention_mask)

        # Recontextualize
        recontextualized_representations = self.recontextualizer(enhanced_span_representations, h_projections, mention_mask)

        # project from knowledge-base back to bert
        h_new = self.dropout(self.kb2bert(recontextualized_representations))
        h_new = self.output_ln(h + h_new)

        # return new hidden state
        return h_new, linking_scores, entropy
