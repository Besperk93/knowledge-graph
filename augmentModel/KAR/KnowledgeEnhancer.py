import torch
import torch.nn as nn



class KnowledgeEnhancer(nn.module):

    def __init__(self, kb, threshold:float=None):
        super(KnowledgeEnhancer, self).__init__()
        self.threshold = threshold
        self.null_embeddings = None if threshold is None else nn.Parameter(torch.zeros(kb.embedding_dimensions))
        self.enhanced_embeddings_ln = nn.LayerNorm(kb.embedding_dimensions)
        set_start_weights(self.enhanced_embeddings_ln, 0.02)

    def forward(self, linking_scores, candidate_embeddings, mention_span_representations, mention_mask):

        if self.threshold is not None:
            below_threshold = linking_scores < self.threshold
            linking_scores.masked_fill(below_threshold, -10000.0)

        normalized_linking_scores = torch.softmax(linking_scores, dim=-1)
        normalized_linking_scores = mention_mask.unsqueeze(-1).float().detach() * normalized_linking_scores

        # compute weighted sum of candidate embeddings
        entity_embeddings = (normalized_linking_scores.unsqueeze(-1) * candidate_embs).sum(-2)
        entity_embeddings = self.dropout(entity_embeddings)

        # handle no value above threshold
        if (self.threshold is not None) and (self.null_emb is not None):
            all_below_threshold = below_threshold.sum(-1) == linking_scores.shape[-1]
            entity_embeddings[all_below_threshold] = self.null_embeddings.unsqueeze(0)

        # compute enhanced span representations
        enhanced_span_representations = mention_span_representations + entity_embeddings
        enhanced_span_representations = self.enhanced_embeddings_ln(enhanced_span_representations)

        # compute entropy loss from linking scores
        probs = normalized_linking_scores[mention_mask]
        log_probs = torch.log(probs + 1e-5)
        entropy = -(probs * log_probs).sum(-1)

        # return enhanced representations and entropy of linking attention
        return enhanced_span_reprs, entropy
