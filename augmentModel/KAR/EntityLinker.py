import torch
import torch.nn as nn

from transformers import BertConfig
from transformers import BertEncoder
from .utilities import set_start_weights, extend_mask


class EntityLinker(nn.module):

    def __init__(self, kb, span_encoder_config:BertConfig, hidden_dimensions:int=100):
        super(EntityLinker, self).__init__()
        # NOTE: keep this as a Bert encoder block as we want to be looking at the full context of any input text, rather than one word at a time etc.
        self.encoder = self._create_span_encoder(kb, span_encoder_config)
        self.score_mlp = nn.Sequential(
            nn.Linear(2, hidden_dimensions),
            nn.RelU(),
            nn.Linear(hidden_dimensions, 1))
        # NOTE: leaving out eps=1e-5 as it's now the default
        self.candidate_embeddings_ln = nn.LayerNorm(kb.embedding_dimensions)
        set_start_weights(self.score_mlp[0], 0.02)
        set_start_weights(self.score_mlp[2], 0.02)
        set_start_weights(self.candidate_embeddings_ln, 0.02)


    def _create_span_encoder(self, kb, span_encoder_config):
        if span_encoder_config is None:
            # NOTE: ndol suggests this will return the identity function as an encoder
            return lambda t, m, h: t
        span_encoder_config.hidden_size = kb.embedding_dimensions
        return BertEncoder(span_encoder_config)


    def forward(self, candidate_embeddings, candidate_mask, candidate_priors, mention_span_representations, mention_span_mask):
        # apply layer norm to candidate embeddings
        candidate_embeddings = self.candidate_embs_ln(candidate_embeddings)
        # apply span-encoder
        # NOTE: if self.encoder is not BERT will head_mask=None be acceptable?
        head_mask = None if not isinstance(self.encoder, BertEncoder) else [None] * self.encoder.config.num_hidden_layers
        extended_mention_span_mask = extend_mask(mention_span_mask)

        mention_span_reprs = self.encoder(
            hidden_states=mention_span_representations,
            attention_mask=extended_mention_span_mask,
            head_mask=head_mask
        )[0]

        # compute entity linking scores
        scores = (candidate_embeddings * mention_span_representations.unsqueeze(-2)).sum(-1) / math.sqrt(candidate_embeddings.size(-1))
        scores_with_prior = torch.cat((scores.unsqueeze(-1), candidate_priors.unsqueeze(-1)), dim=-1)
        linking_scores = self.score_mlp(scores_with_prior).squeeze(-1)
        linking_scores = linking_scores.masked_fill(~candidate_mask.bool(), -10000.0)

        # return linking scores
        return linking_scores
