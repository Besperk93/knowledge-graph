# Adapted from Ndoll98 KnowBert, this module will represent mention spans ready to be passed to the rest of the KAR
import torch
import torch.nn as nn

import math
from .Utilities import set_start_weights


class MentionPooler(nn.Module):

    def __init__(self, dimensions:int):
        super(MentionPooler, self).__init__()
        self.dimensions = dimensions
        # self.attention is a function, initialised as follows
        self.attention = nn.Linear(dimensions, 1, bias=False)
        #NOTE: Initialiser range kept as original repo for now
        set_start_weights(self.attention, 0.02)


    def forward(self, h, spans):
        # h is the tensor representation taken from the previous layer in the model
        # Get shape of input tensor
        shape = h.size()
        # Apply linear function to input tensor
        attention_logits = self.attention(h.view(-1, self.dimensions))
        attention_logits = attention_logits.view(*shape[:-1], 1)
        # gather spans
        idx = torch.arange(spans.size(0)).repeat(spans.size(1), 1).T.unsqueeze(-1)
        # NOTE: What are the elipses doing here?
        sequence_spans = h[idx, spans, ...]
        attention_spans = attention_logits[idx, spans, ...]
        # apply mask and softmax to attention spans
        mask = (spans == -1).unsqueeze(-1)
        attention_spans = attention_spans.masked_fill(mask, -10000)
        attention_weight_spans = torch.softmax(attention_spans, dim=-2)
        attention_weight_spans = attention_weight_spans.masked_fill(mask, 0)
        # compute weighted sum of sequence spans and attention weights
        pooled = (sequence_spans * attention_weight_spans).sum(-2)
        # return pooled tensors
        return pooled



class MentionRepresenter(nn.Module):

    def __init__(self, kb, max_mentions):
        super(MentionRepresenter, self).__init__()
        # Maximum number of entity mentions to look at
        # Note: check original paper
        self.max_mentions = max_mentions
        # Get the dimensions of the embedded knowledge base
        self.embedding_dimensions = kb.embedding_dimensions
        # Establish mention pooler
        self.pooler = MentionPooler(self.embedding_dimensions)
        # Establish linear normalisation layer
        self.span_representation_ln = nn.LayerNorm(self.embedding_dimensions)


    def forward(self, h_projections, mention_spans):
        # Note: Mention Span Representations are known as "C" in the original paper
        # pass projection of h to pooler to create mention span representations
        mention_span_representations = self.pooler(h_projections, mention_spans)
        # Get linear representation of mention spans
        mention_span_representations = self.span_representation_ln(mention_span_representations)
        return mention_span_representations
